//#include <stdlib.h>
//#include <stdio.h>
//#include <ctype.h>
//#include <string.h>
#include <iostream>
#include <queue>
#include <iomanip>
#include <string>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <sm_11_atomic_functions.h>
#include "device_launch_parameters.h"
#include <stdio.h>
using namespace std;

__constant__ int SCCSIZE;
__constant__ int TOTALSIZE;

__constant__ int WARP_T = 32;
__constant__ int MAX_THREAD_SMX = 2048;
__constant__ int MAX_BLOCK_SMX = 16;
__constant__ int MAX_BLOCK_THRESHOLD = 16;
__constant__ int BLOCK_T = 1024;
__constant__ int INITIAL_T;
__constant__ int EXPAND_LEVEL = 2;
__constant__ int BLOCK_SYN_THRESHOLD = 8;
__constant__ int TASKTYPE;
__constant__ bool DEBUG = true; //used for debug infor output

//class Gqueue for global memeory access
class GQueue{
public:
	int ** G_queue;
	//Pathnode ** G_Backup_queue;
	int * G_queue_size;
	//int * G_backup_queue_size; //as a backup
	int blockcount;
	//int backupblockcount;

	GQueue()
	{	//backupblockcount = 0;
	}
	~GQueue(){}
};

class PathNode{
public:
	int presucc;
	int selfid;
};
/***************Global variant****************/
__device__ GQueue G_Queue;
__device__ bool G_ifsccReach;
__device__ int ** P_G_sequence_index; //as a sequencial array to do task partition
__device__ int * P_taskd_index;

//for child use
__device__ int ** C_G_sequence_index; //as a sequencial array to do task partition
__device__ int * C_taskd_index;

//__device__ int * CBackBlockindex; //use as the index to copy back to global memory.
__device__ int * CBackBlockTasksize; //use to record the task size in each block, used for duplicate eliminataion.

__device__ int Child_Expandedtask;
__device__ bool Child_syn_need;
__device__ bool Child_need_back2parent;

__device__ int  Child_block_number;

__device__ int * Child_Queue_index; //used to mark the end write index;
__device__ int * Child_writeback_index; //used to mark the index to start the write/read

__device__ bool * DuplicateEli;
//for the syn between blocks 
__device__ int SynMutex; //for simple syn  
__device__ int * Arrayin;
__device__ int * Arrayout;

__device__ unsigned int G_path2sccmutex;

__device__ int GstartID;
__device__ int Gogwidth;
//how to use const_restrict memory?
__device__ bool G_loopfind;

__device__ bool iffinish;
//syn between blocks
__device__ void __gpu_blocks_simple_syn(int goalval)
{
	//thread ID in a block
	int tid_in_block = threadIdx.x;
	
	// only thread 0 is used for synchronization
	switch(tid_in_block) 
	{
		case 0: 
		{
			atomicAdd(&SynMutex, 1);
			while(SynMutex % goalval != 0) {
			;	}
			break;
		}
		default: break;
	}
	__syncthreads();
}
__device__ void __gpu_blocks_tree_syn(int goalval, int * arrayin, int * arrayout)
{
	// thread ID in a block
	int tid_in_block = threadIdx.x * blockDim.y + threadIdx.y;
	int nBlockNum = gridDim.x * gridDim.y;
	int bid = blockIdx.x * gridDim.y + blockIdx.y;
	
	// only thread 0 is used for synchronization
	if (tid_in_block == 0) 
	{
		Arrayin[bid] = goalval;
	}
	
	if (bid == 1)
	{
		if (tid_in_block < nBlockNum) 
		{
			while (Arrayin[tid_in_block] != goalval) 
			{;}
		}
		__syncthreads();
	
		if (tid_in_block < nBlockNum)
		{
			Arrayout[tid_in_block] = goalval;
		}
	}
	
	if (tid_in_block == 0)
	{
		while (Arrayout[bid] != goalval)
		{ ;}
	}
	__syncthreads();
}

/*******************************************/

//Quick search for if scc reach
__inline__ __device__ bool BSearchIfreach(int *arr,int length,int target)
{
	int left=0;
	int right = length-1;
	while(left <= right){
		int middle = (left+right)/2;
		switch(target == arr[middle]){    //while(rand > degreedis[j])
			case true:return true;
			case false:
				switch(target > arr[middle]){
					case true: left = middle+1; break;
					case false: right = middle-1; break;	
				}
		}
	}
	return false;
}

__global__ void ChildPath(int, int **, int *, int *, int *, int *, int, int *, int * ); 
__global__ void GPath(int startid, int * scc, int * outgoing, int ogwidth, int * path2scc, int * pathrecording, int * G_pathrecMutex)  //for optimization, if outgoing is not very big, it can be stored in the specific memory in kepler
{
	int inblocktid = threadIdx.x;

	int i,j;
	int tmpqcount;
	int QUEUE_AVAI_LENGTH,SUCC_SIZE;
	int tmpnode,peeknode;
	int succ_num = 0;
	int relationindex;
	int writeindex, readindex;
	PathNode precord;
	int inblockwarpnum;
	bool islocalfindscc ;

	int tmpcount = 0;
	int tmp;
	int tmpstart,tmpend;

	int Q_LENGTH;

	__shared__ bool ifinblockadjustment;
	__shared__ int queuesize;
	__shared__ bool ifexpand;
	__shared__ bool ifSccReach;
	__shared__ unsigned int path2sccmutex;
	__shared__ unsigned int ifsccreachmutex;

	__shared__ int Init_S_Queue_head[32];  
	__shared__ int Init_S_Queue_tail[32];
	__shared__ int Init_S_Queue_indexbackup[32];

	__shared__ int S_pathrecord_tail[32]; //no need backup for parent
	__shared__ int inblockexceednum[32];
	__shared__ int inblockavailength[32];

	switch(inblocktid)
	{
		case 0:
		{
			for(i=0;i<32;i++)
			{
				inblockexceednum[j] = 0;
				Init_S_Queue_head[i] = 0;
				Init_S_Queue_tail[i] = 0;
				Init_S_Queue_indexbackup[i] = 0;
				S_pathrecord_tail[i] = 0;
			}

			DuplicateEli = new bool[TOTALSIZE];
			for(i = 0; i < TOTALSIZE; i++)
				DuplicateEli[i] = false;
			ifSccReach = false;
			G_loopfind = false;
			G_path2sccmutex = 0;
		}
		
	}
	

	extern __shared__ int S[];
	
	int * Init_S_Queue[32];
	Init_S_Queue[0] = S;
        for(i = 1; i<WARP_T; i++)
	{
		Init_S_Queue[i]= Init_S_Queue[i-1] + WARP_T;
	}
	
	PathNode * S_Precord_Queue[32];
	S_Precord_Queue[0] = (PathNode *)&Init_S_Queue[31][WARP_T];
	for(i=1; i<WARP_T;i++)
	{
		S_Precord_Queue[i] = &S_Precord_Queue[i-1][WARP_T];
	}
	
	switch(inblocktid)
	{
		case 0:
		{
				for(i = 0; i < WARP_T; i++) 
				{
					for(j = 0; j < WARP_T; j++) 
					{
						Init_S_Queue[i][j] = -1;
					}
				}
		}
		default:break;
	}

	//__syncthreads();

	switch(inblocktid)
	{
		case 0:
		{	
			if(TASKTYPE != 0)
				ifSccReach = BSearchIfreach(scc,SCCSIZE, startid);
		

			switch(ifSccReach)
			{
				case false:
				{
					Init_S_Queue[0][0]=startid;
					Init_S_Queue_tail[0]++; //move head tail, need modification in the S_queue part.
					(S_Precord_Queue[inblocktid][S_pathrecord_tail[inblocktid]]).selfid = startid;
					(S_Precord_Queue[inblocktid][S_pathrecord_tail[inblocktid]++]).presucc = -1;

					queuesize = 1;
					ifexpand = false;
					ifSccReach = false;
					ifinblockadjustment = false;
					path2sccmutex = 0;
					ifsccreachmutex = 0;
				}
			}
		}
	}

	//__syncthreads();
	Q_LENGTH = WARP_T;
	int sccmid;
	int loopcount = 0;
	if(!ifSccReach)
	{
		do{
			loopcount++;
			islocalfindscc = false;
			peeknode = -1;
			switch(Init_S_Queue_tail[inblocktid] != Init_S_Queue_head[inblocktid])
			{
				case true:
				{				
					readindex = Init_S_Queue_head[inblocktid];
					peeknode = Init_S_Queue[inblocktid][readindex];
			
					if(peeknode != -1)
					{
						succ_num = 1;
				
						//judge if belong to scc(sorted)
						if(BSearchIfreach(scc,SCCSIZE, peeknode))
						{
							ifSccReach = true;
							islocalfindscc = true;
							if(loopcount == 1 && TASKTYPE ==0)
							{
								ifSccReach = false;
								islocalfindscc = false;
							}
							iffinish = false;
						}
					
						if(ifSccReach)
						{
							for(i=0;i<S_pathrecord_tail[inblocktid];i++)
							{
								precord = S_Precord_Queue[inblocktid][i];
								if(pathrecording[precord.selfid]!=-1)
									continue;
								else
								{
									if(!atomicExch(&G_pathrecMutex[precord.selfid],1))
									{
										pathrecording[precord.selfid] = precord.presucc;
										//atomicExch(&G_pathrecMutex[precord.selfid],0);
									}
									else
										continue;
								}
							}

							while(!iffinish && islocalfindscc)  
							{  
								if(!atomicExch(&path2sccmutex, 1))   //use lock to modify the path2scc
								{
									path2scc[0] = peeknode;
									relationindex = peeknode;
									for(i=1; pathrecording[relationindex] != startid; i++)
									{
										path2scc[i] = pathrecording[relationindex];
										relationindex = path2scc[i];
									}
									path2scc[i] = startid;
									path2scc[i+1]= -1;
									iffinish = true;
									//atomicExch(&path2sccmutex, 0);
								}
								else
									break;
							}

							break;
						}
					
					}
				}
			}

			if(ifSccReach)
				break;

			switch(peeknode != -1)	
			{
				case true:
				{
					readindex = Init_S_Queue_head[inblocktid];
					writeindex = Init_S_Queue_tail[inblocktid];
					QUEUE_AVAI_LENGTH = Q_LENGTH-((writeindex-readindex+Q_LENGTH) % Q_LENGTH);

					while((outgoing[peeknode*ogwidth+succ_num]) > 0)
					{	
	                 			SUCC_SIZE = outgoing[peeknode*ogwidth];			
						if(SUCC_SIZE < QUEUE_AVAI_LENGTH)
						{
							tmpnode = outgoing[peeknode*ogwidth + succ_num];
							
							(S_Precord_Queue[inblocktid][S_pathrecord_tail[inblocktid]]).selfid = tmpnode;
							(S_Precord_Queue[inblocktid][S_pathrecord_tail[inblocktid]++]).presucc = peeknode;
						
							if(TASKTYPE == 0)
							{
								if(tmpnode == scc[0] && !G_loopfind){
									if(pathrecording[tmpnode] != peeknode)
										pathrecording[tmpnode] = peeknode;
									G_loopfind = true;
								}
							}

							if(S_pathrecord_tail[inblocktid] == 32)
							{
								//queue full,copy back to global memory, here, as the path length is much bigger than the width of graph, so the length of pathrecord queue is important.
								while(S_pathrecord_tail[inblocktid] > 0)
								{
									precord = S_Precord_Queue[inblocktid][--S_pathrecord_tail[inblocktid]];
								
									if(pathrecording[precord.selfid] != -1)
										continue;
									else
									{
										if(!atomicExch(&G_pathrecMutex[precord.selfid], 1))
										{
											pathrecording[precord.selfid]=precord.presucc;
											DuplicateEli[precord.presucc] = true;
											//atomicExch(&G_pathrecMutex[precord.selfid], 0);
										}
									}
								}
								S_pathrecord_tail[inblocktid] = 0;
							}

							writeindex = Init_S_Queue_tail[inblocktid];

							Init_S_Queue[inblocktid][writeindex] = tmpnode;
							Init_S_Queue_tail[inblocktid]++;
							if(Init_S_Queue_tail[inblocktid] == Q_LENGTH)
							{
								
								Init_S_Queue_tail[inblocktid] -= Q_LENGTH;
							}
										
							succ_num++;
						}
						else
						{
							ifinblockadjustment = true;   //if use atomic operation?
							inblockexceednum[inblocktid] = SUCC_SIZE; 
							inblockavailength[inblocktid] = QUEUE_AVAI_LENGTH;
							break;
						}
						inblockexceednum[inblocktid]= succ_num-1-QUEUE_AVAI_LENGTH;  //HOW TO adujustment inblock
						inblockavailength[inblocktid] = QUEUE_AVAI_LENGTH-succ_num-1;

					}

					if(!ifinblockadjustment)
					{
						Init_S_Queue[inblocktid][readindex] = -1;
						if(Init_S_Queue_head[inblocktid]++ == WARP_T)
						{
							Init_S_Queue_head[inblocktid]-= WARP_T;
						}
					}
				}		
			}

			bool ismoreresource = true;
			int avairesourcecount;
			switch(inblocktid)
			{
				case 0:
				{
					if(!ifinblockadjustment)
					{
						for(i = 0;i < WARP_T; i++)
						{	avairesourcecount = 0;
							if(Init_S_Queue_tail[i] == Init_S_Queue_head[i]&& ismoreresource)
							{
								for(j=0; j < WARP_T; j++)
								{
									if((Init_S_Queue_tail[j]-Init_S_Queue_head[j]+WARP_T)%WARP_T > 1)
									{
										avairesourcecount++;
										if((--Init_S_Queue_tail[j]) < 0 )
											Init_S_Queue_tail[j] += Q_LENGTH;

										Init_S_Queue[i][Init_S_Queue_tail[i]++] = Init_S_Queue[j][Init_S_Queue_tail[j]];
										if(Init_S_Queue_tail[i] == Q_LENGTH)
											Init_S_Queue_tail[i] -= Q_LENGTH;
										Init_S_Queue[j][Init_S_Queue_tail[j]] = -1;
										//Init_S_Queue_tail[j] = (Init_S_Queue_tail[j]+WARP_T)%WARP_T;
										if(S_pathrecord_tail[j] >0)
										{
											S_Precord_Queue[i][S_pathrecord_tail[i]++] = S_Precord_Queue[j][--S_pathrecord_tail[j]];
											//if(S_pathrecord_tail[i] == Q_LENGTH)
											//	S_pathrecord_tail[i]-=Q_LENGTH;
										}
										break;
									}
								}
								if(avairesourcecount == 0)
								{
									ismoreresource = false;
									break;
								}
							}
						}	
					}
				}
			}

			//__syncthreads();

			switch(inblocktid)
			{
				case 0:
					{
						queuesize = 0;
						for(i = 0; i < 32; i++)
						{
							queuesize += (Init_S_Queue_tail[i] - Init_S_Queue_head[i] + WARP_T)%WARP_T;
						}
						if(queuesize > INITIAL_T)
						{
							ifexpand = true;
						}
						else
						{					
							if(ifinblockadjustment == true)   //inblock adjustment
							{
								for(i = 0;i<32;i++)
								{	
									succ_num = 1;
									tmpqcount=0;
									if(inblockexceednum[i] > 0)
									{
										readindex = Init_S_Queue_head[i];
										peeknode = Init_S_Queue[i][readindex];
										while((tmpnode=outgoing[tmpnode*ogwidth + succ_num]) != -1)
										{
											while(true)
											{
												if(inblockavailength[tmpqcount] > Q_LENGTH/2) //balance inblock while exceed;
												{
													writeindex = Init_S_Queue_tail[tmpqcount];
													Init_S_Queue[tmpqcount][writeindex] = tmpnode;
													if(Init_S_Queue_tail[tmpqcount]++ == Q_LENGTH)
													{
														Init_S_Queue_tail[tmpqcount] -= WARP_T;
													}
													
													(S_Precord_Queue[tmpqcount][S_pathrecord_tail[tmpqcount]]).selfid = tmpnode;
													(S_Precord_Queue[tmpqcount][S_pathrecord_tail[tmpqcount]++]).presucc = peeknode;

													if(S_pathrecord_tail[tmpqcount] == 32)
                                                        						{
                                                                						//queue full,copy back to global memory, here, as the path length is much bigger than the width of graph, so the length of pathrecord queue is important.
                                                                						while(S_pathrecord_tail[tmpqcount] > 0)
                                                               							 {
                                                                        						precord = S_Precord_Queue[tmpqcount][--S_pathrecord_tail[tmpqcount]];

                                                                        						if(pathrecording[precord.selfid] != -1)
                                                                            							    continue;
                                                                        						else
                                                                        						{
                                                                               							 if(!atomicExch(&G_pathrecMutex[precord.selfid], 1))
                                                                                						{
                                                                                      							  pathrecording[precord.selfid]=precord.presucc;
                                                                                       							  DuplicateEli[precord.presucc] = true;
                                                                                        						//atomicExch(&G_pathrecMutex[precord.selfid], 0);
                                                                                						}
                                                                        						}
                                                                						}
                                                                						S_pathrecord_tail[tmpqcount] = 0;
                                                        						}
														
													succ_num++;
													inblockavailength[tmpqcount]--;
													break;
												}
												else
												{
													tmpqcount++;
													if(tmpqcount == 32)
													{											
														tmpqcount = 0;

													}
												}
											}
										}
								
									}
								}
								queuesize = 0;
								for(i = 0; i < 32; i++)
								{
									queuesize += (Init_S_Queue_tail[i]  - Init_S_Queue_head[i] + WARP_T)%WARP_T;
								}
								if(queuesize > INITIAL_T)
									ifexpand = true;

							}
						}
					}
			}
			//__syncthreads();
		}while(!ifexpand);

		int expandedtasksize = 0;
		int childbsize = 0;  

		switch(!ifSccReach && inblocktid == 0)
		{
			/*!!!important!!!FOR THIS PART, how many task to put in each block is very important, in order to decrease the time to call child, 
			* maybe the thread in each block should be more than the task, this can be verified in experiments*/
			case true:
			{
				expandedtasksize = queuesize;

				if(expandedtasksize % WARP_T == 0)
					childbsize = expandedtasksize/WARP_T;
				else
					childbsize = expandedtasksize/WARP_T + 1;

				G_Queue.G_queue = new int * [childbsize];
				G_Queue.G_queue_size = new int [childbsize];
				G_Queue.blockcount = childbsize;

				for(j=0; j<childbsize; j++)
				{		
					G_Queue.G_queue[j] = new int[TOTALSIZE];
					G_Queue.G_queue_size[j] = 0;
				}
		
				for(i = 0; i < 32; i++)
				{
					while(S_pathrecord_tail[i] > 0)
					{
						precord = S_Precord_Queue[i][--S_pathrecord_tail[i]];
						if(pathrecording[precord.selfid] != -1)
						{
							if(DuplicateEli[precord.presucc] == false)
								DuplicateEli[precord.presucc] = true;
							continue;
						}
						else
						{
							pathrecording[precord.selfid] = precord.presucc;
							if(DuplicateEli[precord.presucc] == false)
								DuplicateEli[precord.presucc] = true;
						}
					}
				
				}

				for(j = 0; j < 32; j++)
				{
					readindex = Init_S_Queue_head[j];
					writeindex = Init_S_Queue_tail[j];
					for(int m = 0; m < ((writeindex - readindex + WARP_T) % WARP_T); m++)
					{
						tmpstart = Init_S_Queue_head[j];
						tmpend = Init_S_Queue_tail[j];
						tmp = Init_S_Queue[j][tmpstart];				
					
						G_Queue.G_queue[tmpcount][G_Queue.G_queue_size[tmpcount]] = Init_S_Queue[j][tmpstart];    //not sure about if the memory copy will work,need confirm.
						G_Queue.G_queue_size[tmpcount]++;

						if(Init_S_Queue_head[j]++ == WARP_T)
							Init_S_Queue_head[j] -= WARP_T;
						tmpcount++;
						tmpcount=tmpcount%(childbsize);
						
					}
				}
			}
		}

		//__syncthreads();
	
		int expandtime = 1;
		int averagetask, lastblocktask;
		int vgcount = 0;
		while(!G_ifsccReach && !ifSccReach)
		{
			//this can be expanded to two version, one is iterative, the other is recursive!
			bool ifneedsyn = false;
			queuesize = 0;

			//rearrange tasks
			switch(inblocktid)
			{
				case 0:
				{
					if(expandtime == 1)
					{
						//add 1 is for the end of the last blocka
						P_taskd_index = new int[childbsize + 1]; 
						P_taskd_index[0] = 0;
						for(i = 0; i < G_Queue.blockcount; i++)
						{
							queuesize += G_Queue.G_queue_size[i];
							P_taskd_index[i+1] = queuesize;
						}
						P_G_sequence_index = new  int * [queuesize];
						for(i=0;i<childbsize;i++)
						{
							vgcount = 0;
							for(j=P_taskd_index[i]; j<P_taskd_index[i+1]; j++)
							{
								P_G_sequence_index[j] = &(G_Queue.G_queue[i][vgcount]);
								vgcount++;
							}
						}
						inblockwarpnum = WARP_T * EXPAND_LEVEL;
					}
					else
					{
						for(i = 0; i < Child_block_number ; i++)
						{
							queuesize += G_Queue.G_queue_size[i];
						}
						expandedtasksize = queuesize;
						if((childbsize=expandedtasksize/WARP_T) < MAX_BLOCK_THRESHOLD)  //here max block threshold is not static, it is based on GPU architecture.
						{
							averagetask = WARP_T;
							if(expandedtasksize % WARP_T == 0)
							{
								childbsize = expandedtasksize/WARP_T;
								lastblocktask = 0;
							}
							else
							{
								childbsize = expandedtasksize/WARP_T + 1;
								lastblocktask = expandedtasksize % WARP_T;
							}
							inblockwarpnum = averagetask * EXPAND_LEVEL;
						}
						else
						{
							averagetask = 2*WARP_T;
							while((childbsize=expandedtasksize/averagetask) > MAX_BLOCK_THRESHOLD)
							{
								averagetask += WARP_T;
							}
							lastblocktask = expandedtasksize % averagetask;
							inblockwarpnum = averagetask + EXPAND_LEVEL*WARP_T;   //it is possible that the warp num exceed the limit.
						}
	
						//free(P_taskd_index);
						P_taskd_index = new int[childbsize + 1];
						for(i=0; i<childbsize;i++)
						{
							P_taskd_index[i] = i*averagetask;
						}
						if(lastblocktask != 0)
							P_taskd_index[i] = (i-1)*averagetask + lastblocktask;
						else
							P_taskd_index[i] = (i)*averagetask;
					
					}
				}
			}
			//__syncthreads();

			////////////////////////////////
	
			switch(inblocktid)  //if add warp in a single block or in mutiple blokcs is needed to eveluate.
			{
				case 0:
					{
						GstartID = startid;
						Gogwidth = ogwidth;
						if(childbsize > 1)
						{
							Arrayin = new int[childbsize];
							Arrayout = new int[childbsize];
							for(i=0; i<childbsize; i++)
							{
								Arrayin[i] = 0;
								Arrayout[i] = 0;
							}
							SynMutex = 0;
							dim3 gridstructure(childbsize,1,1);
							dim3 threadstructure(inblockwarpnum,1,1);
							Child_block_number = childbsize;
							ChildPath<<<gridstructure,threadstructure,10240 * sizeof(int)>>>(GstartID, P_G_sequence_index, P_taskd_index, path2scc,scc,outgoing, ogwidth, pathrecording, G_pathrecMutex);
							cudaDeviceSynchronize();
						}
						else
						{
							Arrayin = new int[1];
							Arrayout = new int[1];
							Arrayin[0] = 0;
							Arrayout[0] = 0;
							SynMutex = 0;
							dim3 gridstructure(1,1,1);
							dim3 threadstructure(128,1,1);
							if((i=INITIAL_T * 4) < WARP_T)
								i = WARP_T;
							Child_block_number = 1;
							ChildPath<<<1,i, 10240*sizeof(int)>>>(GstartID, P_G_sequence_index, P_taskd_index, path2scc,scc,outgoing, Gogwidth,  pathrecording, G_pathrecMutex);
							cudaDeviceSynchronize();
						}
						//call child path,how to combine each block to just one SM?
			
				
						expandedtasksize = 0;
						ifneedsyn = true;
					}
			}
			expandtime++;
			//cudaDeviceSynchronize();

			//if(ifneedsyn)
			//	__syncthreads();
		}	
	}
	//switch(inblocktid)
	//	case 0 : cudaFree(G_Queue.G_queue);
	__syncthreads();
}

__global__ void ChildPath(int GstartID, int ** P_G_sequence_index, int * P_taskd_index, int * p2scc, int * scc, int * outgoing, int Gogwidth, int * pathrecording, int * G_pathrecMutex)   //dynamic parallel in cuda,
{
	int inblocktindex = threadIdx.x;
	int globalthreadindex = blockDim.x * blockIdx.x + threadIdx.x;
	int i,j;
	////////////////////////////
	int TotalBlockTask = 0;
	int MAX_QUEUE_SIZE = WARP_T;  //warp_t 
	int QUEUE_AVAI_LENGTH,SUCC_SIZE; 
	int WARPID, Inwarptid, WARPNum, LWarpTask;  
	bool IFLastHead,IFLastBlock;
	

	int relationindex;  //used for path recording
	int cwriteindex, creadindex;  //used for read/write queue  

	int duration=P_taskd_index[blockIdx.x + 1] - P_taskd_index[blockIdx.x];   //tasks of the whole blokc
	int goalVal = 0; //used for syn among blocks

	int Childpeeknode,tmpnode;
	int succ_num = 1;
	int GBcount;

	bool islocalfindscc ; //juedge if local find

	PathNode cprecord;

	int tmpwcount,tmpqcount;
	WARPID = inblocktindex/WARP_T;
	Inwarptid = inblocktindex%WARP_T;
	if(blockDim.x % WARP_T == 0)
		WARPNum = blockDim.x / WARP_T;
	else
	{
		WARPNum = blockDim.x / WARP_T + 1;
	}

	if(WARPID == WARPNum - 1)
	{
		IFLastHead = true;
	}	
	if(blockIdx.x == blockDim.x - 1)
	{
		IFLastBlock = true;
	}

	__shared__ int BlockQueuesize;

	__shared__ bool ifSccReach;
	__shared__ unsigned int C_path2sccmutex;
	__shared__ unsigned int C_ifsccreachmutex;

	extern __shared__ int C[];
	int * C_Init_S_WarpQueueHead[32];
	
	C_Init_S_WarpQueueHead[0] = C;
	for(i = 1; i < 32; i++)
		C_Init_S_WarpQueueHead[i] = &C_Init_S_WarpQueueHead[i-1][WARPNum];

	int * C_Init_S_WarpQueueTail[32];	
	C_Init_S_WarpQueueTail[0] = &C_Init_S_WarpQueueHead[31][WARPNum];
	for(i = 1; i < 32; i++)
		C_Init_S_WarpQueueTail[i] = &C_Init_S_WarpQueueTail[i-1][WARPNum];
	
	int * C_Warp_Pathtail[32];
	C_Warp_Pathtail[0] = &C_Init_S_WarpQueueTail[31][WARPNum];
	for(i = 1; i < 32; i++)
		C_Warp_Pathtail[i] = &C_Warp_Pathtail[i-1][WARPNum];

	int * C_Warp_Pathhead[32];
	C_Warp_Pathhead[0] = & C_Warp_Pathtail[31][WARPNum];
	for(i = 1; i < 32; i++)
		C_Warp_Pathhead[i] = &C_Warp_Pathhead[i-1][WARPNum];

	int * C_Warp_Pathbackup[32];
	C_Warp_Pathbackup[0] = &C_Warp_Pathhead[31][WARPNum];
	for(i = 1; i < 32; i++)
		C_Warp_Pathbackup[i] = &C_Warp_Pathbackup[i-1][WARPNum+1];

	int * C_Warpqueuebackup[32];
	C_Warpqueuebackup[0] = &C_Warp_Pathbackup[31][WARPNum+1];
	for(i = 1; i < 32; i++)
		C_Warpqueuebackup[i] = &C_Warpqueuebackup[i-1][WARPNum +1];

	if(inblocktindex == 0)
	{
		for(i = 0; i < 32; i++)
		{	for(j = 0; j < WARPNum; j++)
			{
				C_Init_S_WarpQueueHead[i][j] = j*MAX_QUEUE_SIZE;
				C_Init_S_WarpQueueTail[i][j] = j*MAX_QUEUE_SIZE;
				C_Warp_Pathtail[i][j] = j*MAX_QUEUE_SIZE;
				C_Warp_Pathhead[i][j] = j*MAX_QUEUE_SIZE;
				C_Warp_Pathbackup[i][j] = j*MAX_QUEUE_SIZE;
				C_Warpqueuebackup[i][j] = j*MAX_QUEUE_SIZE;
			}
			C_Warp_Pathbackup[i][WARPNum] = j*MAX_QUEUE_SIZE;
			C_Warpqueuebackup[i][WARPNum] = j*MAX_QUEUE_SIZE;
		}
	}
	
	int * Warptasknum = &C_Warpqueuebackup[31][WARPNum+1];
	if(inblocktindex == 0)
	{
		for(i = 1; i < WARPNum; i++)
			Warptasknum[i] = 0;
	}

	int * Inwarpqueuelength[32];
	Inwarpqueuelength[0] = &Warptasknum[WARPNum];
	for(i = 1; i < 32; i++)
		Inwarpqueuelength[i] = &Inwarpqueuelength[i-1][WARPNum];
	
	PathNode * C_Warp_PathRecording[32];
	C_Warp_PathRecording[0] = (PathNode *)&Inwarpqueuelength[31][WARPNum];
	for(i = 1; i < 32; i++)
		C_Warp_PathRecording[i] = &C_Warp_PathRecording[i-1][WARPNum * MAX_QUEUE_SIZE];
	
	int * cpbackindex = (int *)&C_Warp_PathRecording[31][WARPNum*MAX_QUEUE_SIZE];

	int * C_Init_S_WarpQueue[32];
	C_Init_S_WarpQueue[0] = &cpbackindex[WARPNum];
	for(i = 1; i < 32; i++)
		C_Init_S_WarpQueue[i] = &C_Init_S_WarpQueue[i-1][WARPNum*MAX_QUEUE_SIZE];
	
	if(inblocktindex == 0)
	{
		for(i = 0; i<32; i++)
		{
			for(j=0; j<WARPNum * MAX_QUEUE_SIZE; j++)
			{
				C_Init_S_WarpQueue[i][j] = -1;
			}
		}
	}
	bool * ifallwithtask = (bool*)&C_Init_S_WarpQueue[31][WARPNum*MAX_QUEUE_SIZE];
	if(inblocktindex == 0)
	{
		for(i = 0; i < WARPNum; i++)
			ifallwithtask[i] = false;
	}
	bool * ifinblockadjustment = &ifallwithtask[WARPNum];
	if(inblocktindex == 0)
	{
		for(i = 0; i < WARPNum; i++)
			ifinblockadjustment[i] = false;
	}
	bool * nomoreresource = &ifinblockadjustment[WARPNum];
	if(inblocktindex == 0)
	{
		for(i = 0; i < WARPNum; i++)
			nomoreresource[i] = false;
	}
	if(inblocktindex == 0)
	{
		BlockQueuesize = duration;
		ifSccReach = false;
		//ifinblockadjustment = false;
	}
///////////////////////////////////
		
	switch(globalthreadindex)
	{
		case 0:
		{			
			Child_syn_need = false;
			Child_need_back2parent = false;
			Child_Queue_index = new int[gridDim.x];
			Child_writeback_index = new int[gridDim.x];
			Child_Expandedtask = 0;
			CBackBlockTasksize = new int[gridDim.x];
		
			free(G_Queue.G_queue_size);
			G_Queue.G_queue_size = new int[Child_block_number];
			for(i = 0; i < Child_block_number; i++)
				G_Queue.G_queue_size[i] = 0;
		}
	}

	if(gridDim.x < BLOCK_SYN_THRESHOLD)
		__gpu_blocks_simple_syn(gridDim.x);
	else
		__gpu_blocks_tree_syn(goalVal++, Arrayin, Arrayout);
	
	//add warpdecision process

	while(!G_ifsccReach && !Child_need_back2parent)
	{
		//copy data from global memory to shared memory
		duration=P_taskd_index[blockIdx.x + 1] - P_taskd_index[blockIdx.x];

		switch(Inwarptid)  //for lastblock, the tasks is not the multiple of warp_t
		{
			case 0:
				{				
					j = 0;
					for(i=0; i < duration/WARPNum; i++)
					{
						C_Init_S_WarpQueue[j][C_Init_S_WarpQueueTail[j][WARPID]] = *(P_G_sequence_index[P_taskd_index[blockIdx.x] + WARPID * (duration/WARPNum) + i]);
						if((++C_Init_S_WarpQueueTail[j][WARPID]) == C_Warpqueuebackup[j][WARPID+1])
							C_Init_S_WarpQueueTail[j][WARPID] -= MAX_QUEUE_SIZE;
				
						if(j == 31)
							ifallwithtask[WARPID] = true;  //according to the setting of parent, this condition may not happen currently.

						j++;
						j = j % WARP_T;
					}
				}
		}

		switch(Inwarptid == 0 && duration%WARPNum != 0 && IFLastBlock == true)
		{
			case true:
			{		
				LWarpTask = duration - WARPNum*(duration/WARPNum);
				if(WARPID < LWarpTask)
				{	
					C_Init_S_WarpQueue[j][C_Init_S_WarpQueueTail[j][WARPID]] = *(P_G_sequence_index)[P_taskd_index[blockIdx.x] + WARPNum * (duration/WARPNum) + WARPID];
					if((++C_Init_S_WarpQueueTail[j][WARPID]) == C_Warpqueuebackup[j][WARPID+1])
						C_Init_S_WarpQueueTail[j][WARPID] -= MAX_QUEUE_SIZE;
					if(j == 31)
						ifallwithtask[WARPID] = true;
				}
			}
		}
		//__syncthreads();
	
		//if(globalthreadindex == 0)
		//	free(P_G_sequence_index);

		int moreresourcecount;
		while(!ifallwithtask[WARPID] && !ifSccReach &&!nomoreresource[WARPID])  //Initial step, guarantee each queue has task.
		{	
			Childpeeknode = -1;
			if(C_Init_S_WarpQueueTail[Inwarptid][WARPID] != C_Init_S_WarpQueueHead[Inwarptid][WARPID])
			{
				creadindex = C_Init_S_WarpQueueHead[Inwarptid][WARPID];
				Childpeeknode = C_Init_S_WarpQueue[Inwarptid][creadindex];
				if(Childpeeknode!= -1)
				{
					if(BSearchIfreach(scc, SCCSIZE, Childpeeknode))
					{
						ifSccReach = true;
						islocalfindscc = true;
						iffinish = false;
						G_ifsccReach = true;
					}
			 	}
			}
					/*****************SCC/Acc Reach Detect*****************/
			if(gridDim.x < BLOCK_SYN_THRESHOLD)
				__gpu_blocks_simple_syn(gridDim.x);
			else
				__gpu_blocks_tree_syn(goalVal++, Arrayin, Arrayout);

			
			if(ifSccReach)
			{
				
				for(i=C_Warp_Pathhead[Inwarptid][WARPID] ;i<C_Warp_Pathtail[Inwarptid][WARPID];i++)
				{
						cprecord = C_Warp_PathRecording[Inwarptid][i];
						if(pathrecording[cprecord.selfid]!=-1)
							continue;
						else
						{
							if(!atomicExch(&G_pathrecMutex[cprecord.selfid],1))
							{
								pathrecording[cprecord.selfid] = cprecord.presucc;
									//atomicExch(&G_pathrecMutex[cprecord.selfid],0);
							}
							else
								continue;
						}
				}
				__syncthreads();

	
				while(!iffinish && islocalfindscc)  
				{  
					if(!atomicExch(&G_path2sccmutex, 1))   //use lock to modify the path2scc
					{
							
						p2scc[0] = Childpeeknode;
						relationindex = Childpeeknode;
						for(i=1; pathrecording[relationindex] != GstartID; i++)
						{
							p2scc[i] = pathrecording[relationindex];
							relationindex = p2scc[i];
						}
						p2scc[i] = GstartID;
						iffinish = true;
								//atomicExch(&C_path2sccmutex, 0);
								
					}
					else
						break;
				}
					
			}
		
		
			if(ifSccReach || G_ifsccReach)
				break;

			if(Childpeeknode > 0)
			{
					succ_num = 1;
					while(outgoing[Childpeeknode*Gogwidth + succ_num] > 0)
					{
						cwriteindex = C_Init_S_WarpQueueTail[Inwarptid][WARPID];

						tmpnode = outgoing[Childpeeknode*Gogwidth + succ_num];
						(C_Warp_PathRecording[Inwarptid][C_Warp_Pathtail[Inwarptid][WARPID]]).selfid = tmpnode;
						(C_Warp_PathRecording[Inwarptid][C_Warp_Pathtail[Inwarptid][WARPID]++]).presucc = Childpeeknode;

						if(TASKTYPE == 0)
						{
							if(tmpnode == scc[0] && !G_loopfind){
								if(pathrecording[tmpnode] != Childpeeknode)
									pathrecording[tmpnode] = Childpeeknode;
								G_loopfind = true;
							}
						}

						
						if(C_Warp_Pathtail[Inwarptid][WARPID] == C_Warp_Pathhead[Inwarptid][WARPID])
						{
							//queue full,copy back to global memory, here, as the path length is much bigger than the width of graph, so the length of pathrecord queue is important.
							while(C_Warp_Pathtail[Inwarptid][WARPID] != C_Warp_Pathhead[Inwarptid][WARPID])
							{
								cprecord = C_Warp_PathRecording[Inwarptid][--C_Warp_Pathtail[Inwarptid][WARPID]];

								if(pathrecording[cprecord.selfid] != -1)
									continue;
								else
								{
									if(!atomicExch(&G_pathrecMutex[cprecord.selfid], 1))
									{
										pathrecording[cprecord.selfid]=cprecord.presucc;
										DuplicateEli[cprecord.presucc] = true;
										//atomicExch(&G_pathrecMutex[cprecord.selfid], 0);
									}
								}
							}
							cprecord = C_Warp_PathRecording[Inwarptid][C_Warp_Pathhead[Inwarptid][WARPID]];
							if(pathrecording[cprecord.selfid] != -1)
								continue;
							else
							{
								if(!atomicExch(&G_pathrecMutex[cprecord.selfid], 1))
								{
									pathrecording[cprecord.selfid]=cprecord.presucc;
									DuplicateEli[cprecord.presucc] = true;
									//atomicExch(&G_pathrecMutex[cprecord.selfid], 0);
								}
							}
							C_Warp_Pathtail[Inwarptid][WARPID] == C_Warp_Pathbackup[Inwarptid][WARPID];

						}
						C_Init_S_WarpQueue[Inwarptid][cwriteindex] = tmpnode;
						if((++C_Init_S_WarpQueueTail[Inwarptid][WARPID]) == C_Warpqueuebackup[Inwarptid][WARPID + 1])
							C_Init_S_WarpQueueTail[Inwarptid][WARPID] -= MAX_QUEUE_SIZE;
						succ_num++;
					}
					
					if((++C_Init_S_WarpQueueHead[Inwarptid][WARPID]) == C_Warpqueuebackup[Inwarptid][WARPID + 1])
						C_Init_S_WarpQueueHead[Inwarptid][WARPID] -= MAX_QUEUE_SIZE;

			}
			//__syncthreads();

			switch(Inwarptid)
			{
				case 0:
					{	
						for(i = 0;i < 32; i++)
						{
							moreresourcecount = 0;
							if(C_Init_S_WarpQueueTail[i][WARPID] == C_Warp_Pathbackup[i][WARPID] && C_Init_S_WarpQueueTail[i][WARPID] == C_Init_S_WarpQueueHead[i][WARPID])
							{
								for(j=0; j < 32; j++)
								{
									if((C_Init_S_WarpQueueTail[j][WARPID] - C_Init_S_WarpQueueHead[j][WARPID]+MAX_QUEUE_SIZE)%MAX_QUEUE_SIZE > 1)
									{
										moreresourcecount++;			
							
										C_Init_S_WarpQueue[i][C_Init_S_WarpQueueTail[i][WARPID]++] = C_Init_S_WarpQueue[j][--(C_Init_S_WarpQueueTail[j][WARPID])];
										
										if(C_Warp_Pathtail[j][WARPID] != C_Warp_Pathhead[j][WARPID])
											C_Warp_PathRecording[i][C_Warp_Pathtail[i][WARPID]++] = C_Warp_PathRecording[j][--C_Warp_Pathtail[j][WARPID]];
										break;
									}
								}
								if(moreresourcecount == 0)
								{
									nomoreresource[WARPID] = true;
									break;
								}
							}
						}
						if(i == 32)
							ifallwithtask[WARPID] = true;
					}
			}
			//__syncthreads();
		}

		//////////////////////////////////////////////////
		if(!G_ifsccReach)
		{
			creadindex = C_Init_S_WarpQueueHead[Inwarptid][WARPID];
			Childpeeknode = C_Init_S_WarpQueue[Inwarptid][creadindex];
			if(Childpeeknode)
			{
				succ_num = 1;
				if(BSearchIfreach(scc, SCCSIZE, Childpeeknode))
				{
					ifSccReach = true;
					islocalfindscc = true;
					iffinish = false;
					G_ifsccReach =true;
				}
				/*****************SCC/Acc Reach Detect*****************/
				/******************************************/	
			}
			
			if(gridDim.x < BLOCK_SYN_THRESHOLD)
				__gpu_blocks_simple_syn(gridDim.x);
			else
				__gpu_blocks_tree_syn(goalVal++, Arrayin, Arrayout);

			if(ifSccReach)
			{
					
				for(i=C_Warp_Pathbackup[Inwarptid][WARPID];i<C_Warp_Pathtail[Inwarptid][WARPID];i++)
				{
					cprecord = C_Warp_PathRecording[Inwarptid][i];
					if(pathrecording[cprecord.selfid]!=-1)
						continue;
					else
					{
						if(!atomicExch(&G_pathrecMutex[cprecord.selfid],1))
						{
							pathrecording[cprecord.selfid] = cprecord.presucc;
								//atomicExch(&G_pathrecMutex[cprecord.selfid],0);
						}
						else
							continue;
					}
				}
					
				__syncthreads();
				while(!iffinish && islocalfindscc)  
				{  
					switch(!atomicExch(&G_path2sccmutex, 1))   //use lock to modify the path2scc
					{
						case true:
						{
							p2scc[0] = Childpeeknode;
							relationindex = Childpeeknode;
							for(i=1; pathrecording[relationindex] != GstartID; i++)
							{
								p2scc[i] = pathrecording[relationindex];
								relationindex = p2scc[i];
							}

							iffinish = true;
							//atomicExch(&C_path2sccmutex, 0);
						}
					}
				}

				
			}
		}


		if(gridDim.x < BLOCK_SYN_THRESHOLD)
			__gpu_blocks_simple_syn(gridDim.x);
		else
			__gpu_blocks_tree_syn(goalVal++, Arrayin, Arrayout);
		if(ifSccReach || G_ifsccReach)
			break;

		if(Childpeeknode!=-1)
		{
				cwriteindex = C_Init_S_WarpQueueTail[Inwarptid][WARPID];
				//QUEUE_AVAI_LENGTH = MAX_QUEUE_SIZE - (cwriteindex-creadindex+MAX_QUEUE_SIZE)%MAX_QUEUE_SIZE;
				
				//SCCSIZE = outgoing[Childpeeknode][0];

				while(outgoing[Childpeeknode*Gogwidth + succ_num] != -1)
				{
					tmpnode = outgoing[Childpeeknode*Gogwidth + succ_num];
						
					(C_Warp_PathRecording[Inwarptid][C_Warp_Pathtail[Inwarptid][WARPID]]).selfid = tmpnode;
					(C_Warp_PathRecording[Inwarptid][C_Warp_Pathtail[Inwarptid][WARPID]]).presucc = Childpeeknode;
					C_Warp_Pathtail[Inwarptid][WARPID]++;

					if(TASKTYPE == 0)
					{
						if(tmpnode == scc[0] && !G_loopfind){
							if(pathrecording[tmpnode] != Childpeeknode)
								pathrecording[tmpnode] = Childpeeknode;
							G_loopfind = true;
						}
					}


					if(C_Warp_Pathtail[Inwarptid][WARPID] == C_Warp_Pathbackup[Inwarptid][WARPID+1])
					{
						//queue full,copy back to global memory, here, as the path length is much bigger than the width of graph, so the length of pathrecord queue is important.
						while(C_Warp_Pathtail[Inwarptid][WARPID] != C_Warp_Pathhead[Inwarptid][WARPID])
						{
							cprecord = C_Warp_PathRecording[Inwarptid][--C_Warp_Pathtail[Inwarptid][WARPID]];

							if(pathrecording[cprecord.selfid] != -1)
								continue;
							else
							{
								if(!atomicExch(&G_pathrecMutex[cprecord.selfid], 1))
								{
									pathrecording[cprecord.selfid]=cprecord.presucc;
									DuplicateEli[cprecord.presucc] = true;
									//atomicExch(&G_pathrecMutex[cprecord.selfid], 0);
								}
							}
						}
						cprecord = C_Warp_PathRecording[Inwarptid][C_Warp_Pathhead[Inwarptid][WARPID]];
						if(pathrecording[cprecord.selfid] != -1)
							continue;
						else
						{
							if(!atomicExch(&G_pathrecMutex[cprecord.selfid], 1))
							{
								pathrecording[cprecord.selfid]=cprecord.presucc;
								DuplicateEli[cprecord.presucc] = true;
								//atomicExch(&G_pathrecMutex[cprecord.selfid], 0);
							}
						}
						C_Warp_Pathtail[Inwarptid][WARPID] = C_Warp_Pathhead[Inwarptid][WARPID];
					}

					cwriteindex = C_Init_S_WarpQueueTail[Inwarptid][WARPID];
					C_Init_S_WarpQueue[Inwarptid][cwriteindex] = tmpnode;
					
					if((++C_Init_S_WarpQueueTail[Inwarptid][WARPID]) == C_Warpqueuebackup[Inwarptid][WARPID+1])
					{
						C_Init_S_WarpQueueTail[Inwarptid][WARPID] -= MAX_QUEUE_SIZE;
					}

					succ_num++;
				}
				
				if((++C_Init_S_WarpQueueHead[Inwarptid][WARPID]) == C_Warpqueuebackup[Inwarptid][WARPID+1])
				{
					C_Init_S_WarpQueueHead[Inwarptid][WARPID]-= MAX_QUEUE_SIZE;
				}
				Inwarpqueuelength[Inwarptid][WARPID] = (C_Init_S_WarpQueueTail[Inwarptid][WARPID] - C_Init_S_WarpQueueHead[Inwarptid][WARPID]+MAX_QUEUE_SIZE)%MAX_QUEUE_SIZE;
				
		}

		switch(globalthreadindex)
		{
			case 0:	iffinish = false;
		}
		__syncthreads();

		switch(Inwarptid)
		{
			case 0:
				{			
					for(i = 0; i < 32; i++)
						Warptasknum[WARPID] += Inwarpqueuelength[i][WARPID];
					if(Warptasknum[WARPID] > 32)
						ifinblockadjustment[WARPID] = true;
				}
		}
		__syncthreads();

		int indexcount=0;
		int tmp;
		PathNode tmp2;
		switch(inblocktindex)
		{
			case 0:
				{
					for(j = 0; j < WARPNum; j++)
					{
						TotalBlockTask += Warptasknum[j];
						cpbackindex[indexcount]=TotalBlockTask;
						indexcount++;
					}
					if(TotalBlockTask > blockDim.x)
					{
						Child_syn_need = true;
						
					}
					Child_Queue_index[blockIdx.x] = TotalBlockTask;
				}
		}

		if(gridDim.x < BLOCK_SYN_THRESHOLD)
			__gpu_blocks_simple_syn(gridDim.x);
		else
			__gpu_blocks_tree_syn(goalVal++, Arrayin, Arrayout);

		int averagetask, lefttask, tmpmark;
		int tcreadindex;
		int tpcreadindex;//copy back path and tasks

		GBcount = G_Queue.blockcount;
		if(Child_syn_need)
		{
			switch(globalthreadindex)
			{
				case 0:
					{
						for(i=0; i<gridDim.x;i++)
						{
							//Child_Expandedtask += Child_Queue_index[i]; 
							CBackBlockTasksize[i] = Child_Queue_index[i];
						}
				
						for(i = 0; i<GBcount; i++)
						{
							Child_writeback_index[i] = 0;
						}

						for(i = GBcount; i<gridDim.x; i++) //calculate the start index of each block in Gqueue.
						{
							tmpmark = i / (GBcount-1);
							j = i % (GBcount - 1);
					
							Child_Queue_index[i] += Child_Queue_index[j + (tmpmark-1)*GBcount];
							Child_writeback_index[i] = Child_Queue_index[j + (tmpmark-1)*GBcount];
						}
					}
			}

			if(gridDim.x < BLOCK_SYN_THRESHOLD)
				__gpu_blocks_simple_syn(gridDim.x);
			else
				__gpu_blocks_tree_syn(goalVal++, Arrayin, Arrayout);

			if(ifSccReach || G_ifsccReach)
				break;
			
			switch(inblocktindex)
			{
				case 0:
				{
					for(i = 0; i < WARPNum; i++) //copy back path first.
					{
							for(j = 0; j < 32; j++)
							{
								
								while(C_Warp_Pathtail[j][i]!=C_Warp_Pathhead[j][i])
								{
									tpcreadindex = C_Warp_Pathhead[j][i];
									
									cprecord = C_Warp_PathRecording[j][tpcreadindex];
									if(pathrecording[cprecord.selfid] == -1)
									{
										if(!atomicExch(&G_pathrecMutex[cprecord.selfid], 1))
										{
											pathrecording[cprecord.selfid]=cprecord.presucc;
											DuplicateEli[cprecord.presucc] = true;
										//atomicExch(&G_pathrecMutex[cprecord.selfid], 0);
										}
									}
								 	if(++C_Warp_Pathhead[j][i] == C_Warp_Pathbackup[j][i+1])
										C_Warp_Pathhead[j][i] -= MAX_QUEUE_SIZE;
			
								}

							}
						}
						
					

						tmpqcount = 0;

						switch(GBcount)
						{
							case 1: tmp = 0; break;
					
							default:tmp = blockIdx.x % GBcount ;break;
						}

						for(i = 0; i < WARPNum; i++)
						{

							for(j = 0; j < 32; j++)
							{
								while(C_Init_S_WarpQueueHead[j][i] != C_Init_S_WarpQueueTail[j][i])
								{
								
									if(tcreadindex = C_Init_S_WarpQueueHead[j][i])
									tmpnode = C_Init_S_WarpQueue[j][tcreadindex];
									if(DuplicateEli[tmpnode] == false)
									{
										G_Queue.G_queue[tmp][Child_writeback_index[blockIdx.x] + tmpqcount] = tmpnode;
									//G_Queue.G_queue_size[blockIdx.x]++;
										tmpqcount++;
									}
									else
									{
										CBackBlockTasksize[blockIdx.x]--;
									}
									if((++C_Init_S_WarpQueueHead[j][i]) != C_Init_S_WarpQueueTail[j][i] && C_Init_S_WarpQueueHead[j][i]== C_Warpqueuebackup[j][i+1])
									{
										C_Init_S_WarpQueueHead[j][i] -= MAX_QUEUE_SIZE;
									}
								}
							}
							//tmpqcount = tmpqcount;
							//CBackBlockTasksize[blockIdx.x] = tmpqcount;
							
						}
						CBackBlockTasksize[blockIdx.x] = tmpqcount;
						tmpqcount = 0;
					}
				
				default:break;
			}


			if(gridDim.x < BLOCK_SYN_THRESHOLD)
				__gpu_blocks_simple_syn(gridDim.x);
			else
				__gpu_blocks_tree_syn(goalVal++, Arrayin, Arrayout);			
			
			if(inblocktindex == 0)
				G_Queue.G_queue_size[blockIdx.x] = CBackBlockTasksize[blockIdx.x];
			
			switch(globalthreadindex)
			{
				case 0:
					{
						int wholetasknum = 0;
						for(i = 0; i < gridDim.x; i++)
							wholetasknum += CBackBlockTasksize[i];
				
						//P_G_sequence_index= new int * [wholetasknum];
						cudaMalloc((void **)P_G_sequence_index, sizeof(int *)*wholetasknum);
						tmpmark = 0;
						for(i = 0; i < gridDim.x; i++)
						{
							if(GBcount == 1)
								tmp = 0;
							else
								tmp = i%GBcount;

							for(j = 0; j < CBackBlockTasksize[i]; j++)
								P_G_sequence_index[tmpmark + j] = &(G_Queue.G_queue[tmp][Child_writeback_index[i]+j]);
							tmpmark += CBackBlockTasksize[i];
						}
						Child_Expandedtask = wholetasknum;
						if(wholetasknum > (gridDim.x * blockDim.x))
							Child_need_back2parent = true;
						else
						{
							averagetask = Child_Expandedtask/(gridDim.x);
							lefttask = Child_Expandedtask - averagetask*(gridDim.x);
							P_taskd_index[0] = 0;
							for(i=1;i<gridDim.x+1;i++)
							{
								if(lefttask > 0)
								{
									if(i <= lefttask)
										P_taskd_index[i] = P_taskd_index[i-1] + averagetask +1;
									else
										P_taskd_index[i] = P_taskd_index[i-1] + averagetask;
								}
								else
									P_taskd_index[i] = i* averagetask;
							}
						}
					}
			}
			if(!Child_need_back2parent)
			{
					C_Init_S_WarpQueueHead[Inwarptid][WARPID] = C_Warpqueuebackup[Inwarptid][WARPID];
					C_Init_S_WarpQueueTail[Inwarptid][WARPID] = C_Warpqueuebackup[Inwarptid][WARPID];
					C_Warp_Pathtail[Inwarptid][WARPID] = C_Warp_Pathbackup[Inwarptid][WARPID];
					C_Warp_Pathhead[Inwarptid][WARPID] = C_Warp_Pathbackup[Inwarptid][WARPID];
			}
			if(gridDim.x < BLOCK_SYN_THRESHOLD)
				__gpu_blocks_simple_syn(gridDim.x);
			else
				__gpu_blocks_tree_syn(goalVal++, Arrayin, Arrayout);

		}

	}
}


//outtoing array: should add a -1 in the end of each array.
int * CudaPath(int tasktype, int initial_t, int sccsize,  int totalsize, int startID, int * sccnodelist, int * outgoingtransition, int outgoingwidth) //sccnodelist and acceptlist should be sorted for quick search
{	
	int deviceCount;
	int * G_path2scc, *H_path2scc;
	//int * G_path2acc, *H_path2acc;
	int * G_outgoing;
	int * G_sccnodelist;

	int * G_pathrecordingMutex, *H_pathcordingMutex;
	int * G_pathrecording, *H_pathrecording;
	//int * G_acceptlist;
	int i=1;
	size_t acturalsize;

	string returnresult;

	cout<<"sccsize"<<sccsize<<endl;
	
	cudaGetDeviceCount(&deviceCount);
	if(deviceCount == 0)
		return NULL;

	
	cudaDeviceProp prop;
	for(i = 0; i < deviceCount; i++)  
  	{
    	    if(cudaGetDeviceProperties(&prop, i) == cudaSuccess)
    	    {
      	        if(prop.major >= 1)
       		{
         	    break;
        	}
     	    }
   	}
   	if(i == deviceCount)
   	{
     	    return NULL;
   	}
   	cudaDeviceProp sDevProp = prop;
  	cout<<"Device:"<<i<<endl;
  	cout<<"Device name:"<<sDevProp.name<<endl;
   	cout<<"Device memory:"<<sDevProp.totalGlobalMem<<endl;
   	//cout<<"Memory per-block:"<<sDevProp.sharedMemPerBlock<<endl;
   	//cout<<"Register per-block:"<<sDevProp.regsPerBlock<<endl;
   	//cout<<"Warp size:"<<sDevProp.warpSize<<endl;
   	//cout<<"Memory pitch:"<<sDevProp.memPitch<<endl;
   	//cout<<"Constant Memory:"<<sDevProp.totalConstMem<<endl;
   	cout<<"Max thread per-block:"<<sDevProp.maxThreadsPerBlock<<endl;
   	cout<<"Max thread dim:"<<sDevProp.maxThreadsDim[0]<<","<<sDevProp.maxThreadsDim[1]<<","<<sDevProp.maxThreadsDim[2]<<endl;
   	cout<<"Max grid size:"<<sDevProp.maxGridSize[0]<<","<<sDevProp.maxGridSize[1]<<","<<sDevProp.maxGridSize[2]<<endl;
   	//cout<<"Ver:"<<sDevProp.major<<","<<sDevProp.minor<<endl;
   	cout<<"Clock:"<<sDevProp.clockRate<<endl;
   	//cout<<"textureAlignment:"<<sDevProp.textureAlignment<<endl;
   	cudaSetDevice(i);

	cudaMemcpyToSymbol(TASKTYPE, &tasktype, sizeof(int));
 	cudaMemcpyToSymbol(SCCSIZE, &sccsize, sizeof(int));
	cudaMemcpyToSymbol(TOTALSIZE,&totalsize, sizeof(int));
	cudaMemcpyToSymbol(INITIAL_T,&initial_t,sizeof(int));

	//cudasetdevice();  //optional to use
	H_path2scc = new int[totalsize];
	H_pathrecording = new int[totalsize];
	H_pathcordingMutex = new int[totalsize];

	for(i = 0; i < totalsize; i++)
	{
		H_path2scc[i] = -1;
		H_pathrecording[i] = -1;
		H_pathcordingMutex[i] = 0;
	}
	dim3 blockparameterp(initial_t,1,1);
	dim3 gridparameterp(1,1,1);

	
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start,0);//record time

	cudaMalloc((void**)&G_path2scc, sizeof(int)*(totalsize));
	
	cudaMalloc((void**)&G_sccnodelist, sizeof(int)*sccsize);

	cudaMalloc((void**)&G_pathrecording, sizeof(int)*(totalsize));

	cudaMalloc((void**)&G_pathrecordingMutex, sizeof(int)*(totalsize));
	
	cudaMalloc((void**)&G_outgoing, sizeof(int)*outgoingwidth*totalsize);    //outgoing from pat should be a n*m

	cudaMemcpy(G_path2scc,H_path2scc,sizeof(int)*(totalsize),cudaMemcpyHostToDevice);
	
	cudaMemcpy(G_sccnodelist,sccnodelist,sizeof(int)*sccsize, cudaMemcpyHostToDevice);

	cudaMemcpy(G_pathrecording, H_pathrecording, sizeof(int)*(totalsize), cudaMemcpyHostToDevice);
	
	cudaMemcpy(G_pathrecordingMutex, H_pathcordingMutex, sizeof(int)*(totalsize), cudaMemcpyHostToDevice);

	cudaMemcpy(G_outgoing, outgoingtransition, sizeof(int)*outgoingwidth*totalsize, cudaMemcpyHostToDevice);
	GPath<<<gridparameterp,blockparameterp, 5120*sizeof(int)>>>(startID, G_sccnodelist, G_outgoing, outgoingwidth, G_path2scc, G_pathrecording, G_pathrecordingMutex);
	cudaThreadSynchronize();
	cudaMemcpy(H_path2scc,G_path2scc, sizeof(int)*(totalsize), cudaMemcpyDeviceToHost);
	
	cudaEventRecord(stop,0);//record time
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout<<"cuda cost time(ms):"<<elapsedTime<<endl;
	
	cudaFree(G_path2scc);
	cudaFree(G_sccnodelist);
	cudaFree(G_pathrecording);
	cudaFree(G_outgoing);
	//cudaFree(G_Queue.G_queue);

	return H_path2scc;
}

//test ain
int main()
{
	int * outgoing;
	int * outgoingforacc;
	int * sccnlist;
	int * accnlist;
	int * path2scc;
	int * scc2acc;
	int * acc2acc;
	int outgoingwidth;
	int outgoingwidthforacc;
	

	int totalsize;
	int sccsize;
	int accsize;
	int initial_t = 32;
	int startid;
	int i;
	
	ifstream file1;
	ifstream file2;
	ifstream file3;
	ifstream file4;
	ifstream file5;
	ifstream file6;

	ofstream o1;
	ofstream o2;

	file1.open("./expdata/outgoing.txt");
	file2.open("./expdata/outgoingforacc.txt");
	file3.open("./expdata/sccnlist.txt");
	file4.open("./expdata/accnlist.txt");
	file5.open("./expdata/others.txt");
	file6.open("./expdata/Startid.txt");

	//o1.open("./expdata/realisticrelationougoing");
	//o2.open("./expdata/realisticrelationougoingforacc");

	int tmpin;
	int count1,count2;
	file6>>startid;
	file6.close();
	file5>>sccsize;
	file5>>accsize;
	file5>>totalsize;
	file5>>outgoingwidth;
	file5>>outgoingwidthforacc;
	file5.close();

	outgoing = new int[totalsize * outgoingwidth];
	outgoingforacc = new int[totalsize * outgoingwidthforacc];
	sccnlist = new int[sccsize];
	accnlist = new int[accsize];
	count1 = 0;
	count2 = 0;
	while(!file1.eof())
	{
		file1>>tmpin;
	        //o1<<count1<<"->";
		if(tmpin!=-1)
		{
			outgoing[count1*outgoingwidth+count2] = tmpin;
			//o1<<tmpin<<",";
			count2++;
		}
		else
		{
			outgoing[count1*outgoingwidth + count2] = -1;
			count1++;
			count2 = 0;
			//o1<<endl;
		}
	}
	file1.close();
	
	count1=0;
	count2=0;
	while(!file2.eof())
	{
		file2>>tmpin;
		//o2<<count1<<"->";
		if(tmpin!=-1)
		{
			outgoingforacc[count1*outgoingwidthforacc+count2] = tmpin;
			count2++;
			//o2<<tmpin<<",";
		}
		else
		{
			outgoingforacc[count1*outgoingwidthforacc + count2] = -1;
			count1++;
			//o2<<endl;
			count2 = 0;
		}	
	}
	file2.close();

	count1= 0;
	while(!file3.eof())
	{
		file3>>sccnlist[count1];
		count1++;
	}
	file3.close();

	count1=0;
	while(!file4.eof())
	{
		file4>>accnlist[count1];
		count1++;
	}
	file4.close();

	/*outgoingwidth = 4;
	outgoing = new int[127*outgoingwidth];
	sccnlist = new int[2];
	for(int i=0;i<127;i++)
		for(int j=0;j<4;j++)
			outgoing[j*outgoingwidth+ j]=-1;

	sccnlist[0] = 128;
	sccnlist[1] = 129;

	for(int i=0; i<127;i++)
	{
		outgoing[i*outgoingwidth + 0]=2;  // the first position record the amout of succ.
		outgoing[i*outgoingwidth+ 1]=i*2+1;
		outgoing[i*outgoingwidth + 2]=i*2+2;
		outgoing[i*outgoingwidth + 3] = -1;
	}

	path2scc = CudaPath(8,3,255,0,sccnlist,outgoing,4);*/
   	path2scc = CudaPath(1,initial_t, sccsize, totalsize, startid, sccnlist, outgoing, outgoingwidth);
	scc2acc = CudaPath(1,initial_t, accsize, totalsize, path2scc[0], accnlist, outgoingforacc, outgoingwidthforacc);
	if(scc2acc[0] == -1)
		scc2acc[0] = path2scc[0];
	acc2acc = CudaPath(0,initial_t, 1, totalsize, scc2acc[0], &(scc2acc[0]), outgoingforacc, outgoingwidthforacc);

	for(i = 0; path2scc[i]!= -1; i++)
		cout<<path2scc[i]<<" ";

	for(i = 0; scc2acc[i]!=-1; i++)
		cout<<scc2acc[i]<<" ";

	for(i = 0; acc2acc[i]!=-1; i++)
		cout<<acc2acc[i]<<" ";         
	return 1;
}
