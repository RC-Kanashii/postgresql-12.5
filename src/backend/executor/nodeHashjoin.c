/*-------------------------------------------------------------------------
 *
 * nodeHashjoin.c
 *	  Routines to handle hash join nodes
 *
 * Portions Copyright (c) 1996-2019, PostgreSQL Global Development Group
 * Portions Copyright (c) 1994, Regents of the University of California
 *
 *
 * IDENTIFICATION
 *	  src/backend/executor/nodeHashjoin.c
 *
 * PARALLELISM
 *
 * Hash joins can participate in parallel query execution in several ways.  A
 * parallel-oblivious hash join is one where the node is unaware that it is
 * part of a parallel plan.  In this case, a copy of the inner plan is used to
 * build a copy of the hash table in every backend, and the outer plan could
 * either be built from a partial or complete path, so that the results of the
 * hash join are correspondingly either partial or complete.  A parallel-aware
 * hash join is one that behaves differently, coordinating work between
 * backends, and appears as Parallel Hash Join in EXPLAIN output.  A Parallel
 * Hash Join always appears with a Parallel Hash node.
 *
 * Parallel-aware hash joins use the same per-backend state machine to track
 * progress through the hash join algorithm as parallel-oblivious hash joins.
 * In a parallel-aware hash join, there is also a shared state machine that
 * co-operating backends use to synchronize their local state machines and
 * program counters.  The shared state machine is managed with a Barrier IPC
 * primitive.  When all attached participants arrive at a barrier, the phase
 * advances and all waiting participants are released.
 *
 * When a participant begins working on a parallel hash join, it must first
 * figure out how much progress has already been made, because participants
 * don't wait for each other to begin.  For this reason there are switch
 * statements at key points in the code where we have to synchronize our local
 * state machine with the phase, and then jump to the correct part of the
 * algorithm so that we can get started.
 *
 * One barrier called build_barrier is used to coordinate the hashing phases.
 * The phase is represented by an integer which begins at zero and increments
 * one by one, but in the code it is referred to by symbolic names as follows:
 *
 *   PHJ_BUILD_ELECTING              -- initial state
 *   PHJ_BUILD_ALLOCATING            -- one sets up the batches and table 0
 *   PHJ_BUILD_HASHING_INNER         -- all hash the inner rel
 *   PHJ_BUILD_HASHING_OUTER         -- (multi-batch only) all hash the outer
 *   PHJ_BUILD_DONE                  -- building done, probing can begin
 *
 * While in the phase PHJ_BUILD_HASHING_INNER a separate pair of barriers may
 * be used repeatedly as required to coordinate expansions in the number of
 * batches or buckets.  Their phases are as follows:
 *
 *   PHJ_GROW_BATCHES_ELECTING       -- initial state
 *   PHJ_GROW_BATCHES_ALLOCATING     -- one allocates new batches
 *   PHJ_GROW_BATCHES_REPARTITIONING -- all repartition
 *   PHJ_GROW_BATCHES_FINISHING      -- one cleans up, detects skew
 *
 *   PHJ_GROW_BUCKETS_ELECTING       -- initial state
 *   PHJ_GROW_BUCKETS_ALLOCATING     -- one allocates new buckets
 *   PHJ_GROW_BUCKETS_REINSERTING    -- all insert tuples
 *
 * If the planner got the number of batches and buckets right, those won't be
 * necessary, but on the other hand we might finish up needing to expand the
 * buckets or batches multiple times while hashing the inner relation to stay
 * within our memory budget and load factor target.  For that reason it's a
 * separate pair of barriers using circular phases.
 *
 * The PHJ_BUILD_HASHING_OUTER phase is required only for multi-batch joins,
 * because we need to divide the outer relation into batches up front in order
 * to be able to process batches entirely independently.  In contrast, the
 * parallel-oblivious algorithm simply throws tuples 'forward' to 'later'
 * batches whenever it encounters them while scanning and probing, which it
 * can do because it processes batches in serial order.
 *
 * Once PHJ_BUILD_DONE is reached, backends then split up and process
 * different batches, or gang up and work together on probing batches if there
 * aren't enough to go around.  For each batch there is a separate barrier
 * with the following phases:
 *
 *  PHJ_BATCH_ELECTING       -- initial state
 *  PHJ_BATCH_ALLOCATING     -- one allocates buckets
 *  PHJ_BATCH_LOADING        -- all load the hash table from disk
 *  PHJ_BATCH_PROBING        -- all probe
 *  PHJ_BATCH_DONE           -- end
 *
 * Batch 0 is a special case, because it starts out in phase
 * PHJ_BATCH_PROBING; populating batch 0's hash table is done during
 * PHJ_BUILD_HASHING_INNER so we can skip loading.
 *
 * Initially we try to plan for a single-batch hash join using the combined
 * work_mem of all participants to create a large shared hash table.  If that
 * turns out either at planning or execution time to be impossible then we
 * fall back to regular work_mem sized hash tables.
 *
 * To avoid deadlocks, we never wait for any barrier unless it is known that
 * all other backends attached to it are actively executing the node or have
 * already arrived.  Practically, that means that we never return a tuple
 * while attached to a barrier, unless the barrier has reached its final
 * state.  In the slightly special case of the per-batch barrier, we return
 * tuples while in PHJ_BATCH_PROBING phase, but that's OK because we use
 * BarrierArriveAndDetach() to advance it to PHJ_BATCH_DONE without waiting.
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"

#include "access/htup_details.h"
#include "access/parallel.h"
#include "executor/executor.h"
#include "executor/hashjoin.h"
#include "executor/nodeHash.h"
#include "executor/nodeHashjoin.h"
#include "miscadmin.h"
#include "pgstat.h"
#include "utils/memutils.h"
#include "utils/sharedtuplestore.h"


/*
 * States of the ExecHashJoin state machine
 */
#define HJ_BUILD_HASHTABLE		1
#define HJ_NEED_NEW_OUTER		2
#define HJ_SCAN_BUCKET			3
#define HJ_FILL_OUTER_TUPLE		4
#define HJ_FILL_INNER_TUPLES	5
#define HJ_NEED_NEW_BATCH		6
// 新增
#define HJ_NEED_NEW_INNER		7
#define HJ_GET_AND_HASH_TUPLE   8

/* Returns true if doing null-fill on outer relation */
#define HJ_FILL_OUTER(hjstate)	((hjstate)->hj_NullInnerTupleSlot != NULL)
/* Returns true if doing null-fill on inner relation */
#define HJ_FILL_INNER(hjstate)	((hjstate)->hj_NullOuterTupleSlot != NULL)

static TupleTableSlot *ExecHashJoinOuterGetTuple(PlanState *outerNode,
												 HashJoinState *hjstate,
												 uint32 *outerHashvalue);
static TupleTableSlot *ExecParallelHashJoinOuterGetTuple(PlanState *outerNode,
														 HashJoinState *hjstate,
														 uint32 *outerHashvalue);
static TupleTableSlot *ExecHashJoinGetSavedTuple(HashJoinState *hjstate,
												 BufFile *file,
												 uint32 *outerHashvalue,
												 TupleTableSlot *tupleSlot);
static bool ExecHashJoinNewBatch(HashJoinState *hjstate);
static bool ExecParallelHashJoinNewBatch(HashJoinState *hjstate);
static void ExecParallelHashJoinPartitionOuter(HashJoinState *node);

// 新增
static TupleTableSlot *ExecGetTuple(HashState *hashNode, HashJoinState * hjstate);


/* ----------------------------------------------------------------
 *		ExecHashJoinImpl
 *
 *		This function implements the Hybrid Hashjoin algorithm.  It is marked
 *		with an always-inline attribute so that ExecHashJoin() and
 *		ExecParallelHashJoin() can inline it.  Compilers that respect the
 *		attribute should create versions specialized for parallel == true and
 *		parallel == false with unnecessary branches removed.
 *
 *		Note: the relation we build hash table on is the "inner"
 *			  the other one is "outer".
 * ----------------------------------------------------------------
 */

static pg_attribute_always_inline TupleTableSlot *
// PlanState *pstate：指向当前计划节点的状态
// bool parallel：一个布尔值，指示是否以并行方式执行哈希连接
ExecHashJoinImpl(PlanState *pstate, bool parallel)
{
	HashJoinState *node = castNode(HashJoinState, pstate);  // 将 pstate 转换为 HashJoinState 类型的指针
	// PlanState  *outerNode;  // 外部计划状态
	// HashState  *hashNode;  // 内部计划状态
	ExprState  *joinqual;  // 连接条件
	ExprState  *otherqual;  // 其他条件
	ExprContext *econtext;  // 表达式上下文

	// 新增的哈希表
	HashJoinTable innerHashtable; // 内表哈希表
	HashJoinTable outerHashtable; // 外表哈希表

	// 新增的元组槽
	TupleTableSlot *innerTupleSlot; // 内表元组槽
	TupleTableSlot *outerTupleSlot; // 外表元组槽

	// 新增的哈希值
	uint32		outerHashvalue;
	uint32		innerHashvalue;

	int			batchno = 0;
	// ParallelHashJoinState *parallel_state;  // 并行状态

	// 新增的 HashState
	HashState *innerHashNode; // 内表哈希节点
	HashState *outerHashNode; // 外表哈希节点

	// 新增的元组
	TupleTableSlot *inntuple;
	TupleTableSlot *outtuple;
	HeapTuple	curtuple;

	// 获取哈希节点的 hashEcontext，注意不要和 HashNodeJoin 的 econtext 混淆
	ExprContext *hashEcontext;

	/*
	 * get information from HashJoin node
	 */
	joinqual = node->js.joinqual;
	otherqual = node->js.ps.qual;

	// 获取左哈希节点和右哈希节点
	innerHashNode = (HashState *) innerPlanState(node);
	outerHashNode = (HashState *) outerPlanState(node);

	// 获取左右节点的哈希表
	outerHashtable = node->hj_OuterHashTable;
	innerHashtable = node->hj_InnerHashTable;

	econtext = node->js.ps.ps_ExprContext;

	// parallel_state = outerHashNode->parallel_state; // 使用的是外表的状态

	// 如果是内连接并且有已经匹配的外表元组，可以直接跳过内表的探测过程
	// if (node->js.jointype == JOIN_INNER && node->hj_MatchedOuter)
	// 	node->hj_NeedNewOuterTuple = true;

	/*
	 * Reset per-tuple memory context to free any expression evaluation
	 * storage allocated in the previous tuple cycle.
	 */
	ResetExprContext(econtext);

	// hashEcontext 清理
	ResetExprContext(innerHashNode->ps.ps_ExprContext);
	ResetExprContext(outerHashNode->ps.ps_ExprContext);

	/*
	 * run the hash join state machine
	 */
	for (;;)
	{
		/*
		 * It's possible to iterate this loop many times before returning a
		 * tuple, in some pathological cases such as needing to move much of
		 * the current batch to a later batch.  So let's check for interrupts
		 * each time through.
		 */
		CHECK_FOR_INTERRUPTS();

		switch (node->hj_JoinState)
		{
			case HJ_BUILD_HASHTABLE:

				/*
				 * First time through: build hash table for inner relation.
				 */
				// Assert(hashtable == NULL);  // 确保这是首次构建哈希表，哈希表变量应为NULL

				/*
				 * If the outer relation is completely empty, and it's not
				 * right/full join, we can quit without building the hash
				 * table.  However, for an inner join it is only a win to
				 * check this when the outer relation's startup cost is less
				 * than the projected cost of building the hash table.
				 * Otherwise it's best to build the hash table first and see
				 * if the inner relation is empty.  (When it's a left join, we
				 * should always make this check, since we aren't going to be
				 * able to skip the join on the strength of an empty inner
				 * relation anyway.)
				 *
				 * If we are rescanning the join, we make use of information
				 * gained on the previous scan: don't bother to try the
				 * prefetch if the previous scan found the outer relation
				 * nonempty. This is not 100% reliable since with new
				 * parameters the outer relation might yield different
				 * results, but it's a good heuristic.
				 *
				 * The only way to make the check is to try to fetch a tuple
				 * from the outer plan node.  If we succeed, we have to stash
				 * it away for later consumption by ExecHashJoinOuterGetTuple.
				 */
				// 现在只考虑
				// if (HJ_FILL_INNER(node))
				// {
				// 	/* no chance to not build the hash table */
				// 	// 如果是右连接，总是需要构建哈希表，因此不尝试早期退出
				// 	node->hj_FirstOuterTupleSlot = NULL;
				// }
				// else if (HJ_FILL_OUTER(node) ||
				// 		 (outerNode->plan->startup_cost < hashNode->ps.plan->total_cost &&
				// 		  !node->hj_OuterNotEmpty))
				// {
				// 	// 尝试从外部节点获取一个元组
				// 	node->hj_FirstOuterTupleSlot = ExecProcNode(outerNode);
				// 	if (TupIsNull(node->hj_FirstOuterTupleSlot))
				// 	{
				// 		// 如果外部节点为空，标记外部非空为假
				// 		node->hj_OuterNotEmpty = false;
				// 		return NULL;
				// 	}
				// 	else
				// 		node->hj_OuterNotEmpty = true;
				// }
				// else
				// 	// 不尝试获取外部元组
				// 	node->hj_FirstOuterTupleSlot = NULL;

				/*
				 * Create the hash table.  If using Parallel Hash, then
				 * whoever gets here first will create the hash table and any
				 * later arrivals will merely attach to it.
				 */
				// 创建哈希表，根据是否需要处理空值和连接类型来决定
				// 构建外表的哈希表
				if (outerHashtable == NULL) {
					outerHashtable = ExecHashTableCreate(outerHashNode, node->hj_HashOperators, node->hj_Collations, HJ_FILL_INNER(node));
					node->hj_OuterHashTable = outerHashtable;
					outerHashNode->hashtable = outerHashtable;
				}

				// 构建内表的哈希表
				if (innerHashtable == NULL) {
					innerHashtable = ExecHashTableCreate(innerHashNode, node->hj_HashOperators, node->hj_Collations, HJ_FILL_OUTER(node));
					node->hj_InnerHashTable = innerHashtable;
					innerHashNode->hashtable = innerHashtable;
				}

				/*
				 * Execute the Hash node, to build the hash table.  If using
				 * Parallel Hash, then we'll try to help hashing unless we
				 * arrived too late.
				 */
				// 执行哈希节点，构建哈希表
				// 由于对称哈希一次只哈希一条元组，因此不能使用 MultiExecProcNode 生成全部的哈希表
				// (void) MultiExecProcNode((PlanState *) hashNode);

				

				/*
				 * If the inner relation is completely empty, and we're not
				 * doing a left outer join, we can quit without scanning the
				 * outer relation.
				 */
				// if (hashtable->totalTuples == 0 && !HJ_FILL_OUTER(node))
				// 	return NULL;

				/*
				 * need to remember whether nbatch has increased since we
				 * began scanning the outer relation
				 */
				// 记录当前批次，以便之后检查是否增加
				// 不用考虑批次
				// hashtable->nbatch_outstart = hashtable->nbatch;

				// node->hj_JoinState = HJ_NEED_NEW_OUTER;  // 设置下一个状态，准备处理外部元组
				// 新状态：使用另一个哈希函数 f 计算哈希值，并且在另一张表进行探测
				node->hj_JoinState = HJ_GET_AND_HASH_TUPLE;

				/* FALL THRU */
			case HJ_GET_AND_HASH_TUPLE:

				// 先判断内外表是否耗尽
				if (node->hj_InnerEnd && node->hj_OuterEnd) {
					// 说明没有匹配结果
					return NULL;
				}

				// 内表耗尽，直接扫描外表
				if (node->hj_InnerEnd) {
					outerTupleSlot = ExecGetTuple(outerHashNode, node);
					if (TupIsNull(outerTupleSlot)) {
						// 外表也耗尽了
						node->hj_OuterEnd = true;
						node->hj_JoinState = HJ_GET_AND_HASH_TUPLE;
						continue;
					}
					// 存放在 hj_OuterCurTuple
					node->hj_OuterCurTuple = outerTupleSlot;
					node->hj_JoinState = HJ_SCAN_BUCKET;
					continue;
				}

				// 外表耗尽，直接扫描内表
				if (node->hj_OuterEnd) {
					innerTupleSlot = ExecGetTuple(innerHashNode, node);
					if (TupIsNull(innerTupleSlot)) {
						// 内表也耗尽了
						node->hj_InnerEnd = true;
						node->hj_JoinState = HJ_GET_AND_HASH_TUPLE;
						continue;
					}
					// 存放在 hj_InnerCurTuple
					node->hj_InnerCurTuple = innerTupleSlot;
					node->hj_JoinState = HJ_SCAN_BUCKET;
					continue;
				}
				

				// 从哈希节点获取元组，并顺便哈希后插入到对应的哈希表
				// 需要判断是探测 inner 还是 outer
				if (node->hj_FetchingFromInner) {
					// 存放被哈希的内表元组
					innerTupleSlot = ExecProcNode((PlanState *) innerHashNode);
					// 检查内表是否已经耗尽
					if (TupIsNull(innerTupleSlot)) {
						node->hj_InnerEnd = true;
						// 转去扫描外表
						node->hj_FetchingFromInner = false;
						node->hj_JoinState = HJ_GET_AND_HASH_TUPLE;
						continue;
					}
					// 存放在 hj_InnerCurTuple
					// node->hj_InnerCurTuple = innerTupleSlot;
				} else {
					// 存放被哈希的外表元组
					outerTupleSlot = ExecProcNode((PlanState *) outerHashNode);
					// 检查内表是否已经耗尽
					if (TupIsNull(outerTupleSlot)) {
						node->hj_OuterEnd = true;
						// 转去扫描内表
						node->hj_FetchingFromInner = true;
						node->hj_JoinState = HJ_GET_AND_HASH_TUPLE;
						continue;
					}
					// 存放在 hj_OuterCurTuple
					// node->hj_OuterCurTuple = outerTupleSlot;
				}

				/*
				 * Reset OuterNotEmpty for scan.  (It's OK if we fetched a
				 * tuple above, because ExecHashJoinOuterGetTuple will
				 * immediately set it again.)
				 */
				/*
	             * 重置 OuterNotEmpty 状态以进行扫描。
	             * (如果上面成功提取了一个元组，因为 ExecHashJoinOuterGetTuple 会立即再次设置它。)
	             */
				node->hj_OuterNotEmpty = false;
				node->hj_InnerNotEmpty = false; // 新增

				// 使用另一个哈希函数 f 计算哈希值，并且在另一张表进行探测
				node->hj_JoinState = HJ_SCAN_BUCKET;

				// FALL THROUGH

			case HJ_NEED_NEW_OUTER:

				// /*
				//  * We don't have an outer tuple, try to get the next one
				//  */
				// // 如果没有外部元组，尝试获取下一个
				// outerTupleSlot =
				// 	ExecHashJoinOuterGetTuple(outerNode, node, &hashvalue);
				// // 如果外部元组为空，表示批处理结束或可能整个连接结束
				// if (TupIsNull(outerTupleSlot))
				// {
				// 	/* end of batch, or maybe whole join */
				// 	if (HJ_FILL_INNER(node))
				// 	{
				// 		/* set up to scan for unmatched inner tuples */
				// 		// 准备扫描未匹配的内部元组
				// 		ExecPrepHashTableForUnmatched(node);
				// 		// 切换到扫描未匹配内部元组的状态
				// 		node->hj_JoinState = HJ_FILL_INNER_TUPLES;
				// 	}
				// 	else
				// 		// 切换到需要获取新批次的状态
				// 		node->hj_JoinState = HJ_NEED_NEW_BATCH;
				// 	continue;
				// }

				// // 设置外部元组到执行上下文中
				// econtext->ecxt_outertuple = outerTupleSlot;
				// // 标记当前外部元组还未匹配
				// node->hj_MatchedOuter = false;

				// /*
				//  * Find the corresponding bucket for this tuple in the main
				//  * hash table or skew hash table.
				//  */
				// // 找到当前元组对应的主哈希表或倾斜哈希表中的桶
				// node->hj_OuterCurHashValue = hashvalue;
				// // 获取当前元组所属的桶编号和批次编号
				// ExecHashGetBucketAndBatch(hashtable, hashvalue,
				// 						  &node->hj_OuterCurBucketNo, &batchno);
				// // 获取当前元组所属的倾斜桶编号（如果有）
				// node->hj_OuterCurSkewBucketNo = ExecHashGetSkewBucket(hashtable,
				// 												 hashvalue);
				// node->hj_OuterCurTuple = NULL;

				// /*
				//  * The tuple might not belong to the current batch (where
				//  * "current batch" includes the skew buckets if any).
				//  */
				// // 如果当前元组所属的批次编号不是当前批次，并且不属于任何倾斜桶
				// if (batchno != hashtable->curbatch &&
				// 	node->hj_OuterCurSkewBucketNo == INVALID_SKEW_BUCKET_NO)
				// {
				// 	bool		shouldFree;
				// 	// 提取外部元组的最小元组形式
				// 	MinimalTuple mintuple = ExecFetchSlotMinimalTuple(outerTupleSlot,
				// 													  &shouldFree);

				// 	/*
				// 	 * Need to postpone this outer tuple to a later batch.
				// 	 * Save it in the corresponding outer-batch file.
				// 	 */
				// 	/*
                //      * 需要将这个外部元组推迟到后续批次。
                //      * 将其保存到对应的外部批次文件中。
                //      */
				// 	// Assert(parallel_state == NULL);
				// 	Assert(batchno > hashtable->curbatch);
				// 	// 提取外部元组的最小元组形式	
				// 	ExecHashJoinSaveTuple(mintuple, hashvalue,
				// 						  &hashtable->outerBatchFile[batchno]);

				// 	if (shouldFree)
				// 		heap_free_minimal_tuple(mintuple);

				// 	/* Loop around, staying in HJ_NEED_NEW_OUTER state */
				// 	// 继续循环，仍然保持在 HJ_NEED_NEW_OUTER 状态
				// 	continue;
				// }

				// /* OK, let's scan the bucket for matches */
				// /* 如果当前元组属于当前批次，切换状态为扫描桶以查找匹配项 */
				// node->hj_JoinState = HJ_SCAN_BUCKET;

				/* FALL THRU */

			case HJ_SCAN_BUCKET:

				if (node->hj_FetchingFromInner) {
					// 使用外表的econtext、哈希表和哈希函数
					hashEcontext = outerHashNode->ps.ps_ExprContext;
					// 注意要哈希的元组必须放在 econtext->ecxt_outertuple
					// ecxt_outertuple 要转换成 MinimalTupleSlot
					hashEcontext->ecxt_outertuple = MakeTupleTableSlot(ExecGetResultType((PlanState *)innerHashNode), &TTSOpsMinimalTuple);
					// ExecForceStoreMinimalTuple(innerTupleSlot->tts_ops->copy_minimal_tuple, hashEcontext->ecxt_outertuple, false);
					// hashEcontext->ecxt_outertuple = ExecStoreMinimalTuple(HJTUPLE_MINTUPLE(innerTupleSlot), node->hj_InnerHashTupleSlot, false);
					hashEcontext->ecxt_outertuple = ExecStoreMinimalTuple(ExecCopySlotMinimalTuple(innerTupleSlot), hashEcontext->ecxt_outertuple, false);
					slot_getallattrs(hashEcontext->ecxt_outertuple);
					hashEcontext->ecxt_innertuple = hashEcontext->ecxt_outertuple;
					
					// TTSOpsMinimalTuple 可以通过检查
					// hashEcontext->ecxt_innertuple = MakeTupleTableSlot(ExecGetResultType((PlanState *)innerHashNode), &TTSOpsMinimalTuple);
					// ExecForceStoreMinimalTuple(innerTupleSlot->tts_ops->copy_minimal_tuple, hashEcontext->ecxt_innertuple, false);
					
					ExecHashGetHashValue(outerHashtable,
										 hashEcontext,
										 node->hj_InnerHashKeys,
										 false,
										 false,
										 &innerHashvalue);
					// ResetExprContext(econtext);
					// node->hj_InnerCurTuple = innerTupleSlot;
					// 内表元组经过外表哈希函数处理后的哈希值
					node->hj_InnerCurHashValue = innerHashvalue;
					// 求内表元组在外表哈希表中对应的 bucketNo
					ExecHashGetBucketAndBatch(outerHashtable,
											  innerHashvalue,
											  &node->hj_InnerCurBucketNo,
											  &batchno);
				} else {
					// 使用内表的econtext、哈希表和哈希函数
					hashEcontext = innerHashNode->ps.ps_ExprContext;
					// 注意要哈希的元组必须放在 econtext->ecxt_outertuple
					// ecxt_outertuple 要转换成 MinimalTupleSlot
					hashEcontext->ecxt_outertuple = MakeTupleTableSlot(ExecGetResultType((PlanState *)outerHashNode), &TTSOpsMinimalTuple);
					// ExecForceStoreMinimalTuple(outerTupleSlot->tts_ops->copy_minimal_tuple, hashEcontext->ecxt_outertuple, false);
					// hashEcontext->ecxt_outertuple = ExecStoreMinimalTuple(HJTUPLE_MINTUPLE(outerTupleSlot), node->hj_OuterHashTupleSlot, false);
					hashEcontext->ecxt_outertuple = ExecStoreMinimalTuple(ExecCopySlotMinimalTuple(outerTupleSlot), hashEcontext->ecxt_outertuple, false);
					slot_getallattrs(hashEcontext->ecxt_outertuple);
					hashEcontext->ecxt_innertuple = hashEcontext->ecxt_outertuple;
					
					// TTSOpsMinimalTuple 可以通过检查
					// hashEcontext->ecxt_innertuple = MakeTupleTableSlot(ExecGetResultType((PlanState *)outerHashNode), &TTSOpsMinimalTuple);
					// ExecForceStoreMinimalTuple(outerTupleSlot->tts_ops->copy_minimal_tuple, hashEcontext->ecxt_innertuple, false);
					
					ExecHashGetHashValue(innerHashtable,
										 hashEcontext,
										 node->hj_OuterHashKeys,
										 false,
										 false,
										 &outerHashvalue);
					// ResetExprContext(econtext);
					// node->hj_OuterCurTuple = outerTupleSlot;
					// 外表元组经过内表哈希函数处理后的哈希值
					node->hj_OuterCurHashValue = outerHashvalue;
					// 求外表元组在内表哈希表中对应的 bucketNo
					ExecHashGetBucketAndBatch(innerHashtable,
											  outerHashvalue,
											  &node->hj_OuterCurBucketNo,
											  &batchno);
				}
				/*
				 * Scan the selected hash bucket for matches to current outer
				 */
				// 执行哈希桶扫描。如果当前外部元组没有匹配的内部元组，则切换到处理外部连接的填充元组状态

				// 要事先把 hashEcontext->ecxt_outertuple 转移到 econtext->ecxt_outertuple 中
				// econtext->ecxt_outertuple = hashEcontext->ecxt_outertuple;
				if (node->hj_FetchingFromInner) {
					// econtext->ecxt_outertuple = node->hj_InnerCurTuple;
					// econtext->ecxt_outertuple = MakeTupleTableSlot(ExecGetResultType((PlanState *)innerHashNode), &TTSOpsMinimalTuple);
					// ExecForceStoreMinimalTuple(innerTupleSlot->tts_ops->copy_minimal_tuple, econtext->ecxt_outertuple, false);
					// ExecStoreMinimalTuple(ExecCopySlotMinimalTuple(innerTupleSlot), econtext->ecxt_outertuple, false);
					// econtext->ecxt_outertuple = ExecStoreMinimalTuple(HJTUPLE_MINTUPLE(innerTupleSlot), node->hj_InnerHashTupleSlot, false);
					econtext->ecxt_outertuple = hashEcontext->ecxt_outertuple;
				} else {
					// econtext->ecxt_outertuple = node->hj_OuterCurTuple;
					// econtext->ecxt_outertuple = MakeTupleTableSlot(ExecGetResultType((PlanState *)outerHashNode), &TTSOpsMinimalTuple);
					// ExecForceStoreMinimalTuple(outerTupleSlot->tts_ops->copy_minimal_tuple, econtext->ecxt_outertuple, false);
					// ExecStoreMinimalTuple(ExecCopySlotMinimalTuple(outerTupleSlot), econtext->ecxt_outertuple, false);
					// econtext->ecxt_outertuple = ExecStoreMinimalTuple(HJTUPLE_MINTUPLE(outerTupleSlot), node->hj_OuterHashTupleSlot, false);
					econtext->ecxt_outertuple = hashEcontext->ecxt_outertuple;
				}
				slot_getallattrs(econtext->ecxt_outertuple);

				// 清空 CurTuple
				node->hj_InnerCurTuple = NULL;
				node->hj_OuterCurTuple = NULL;

				// 使用的是另一张表的哈希函数进行探测
				// 结果在 econtext->ecxt_innertuple 或者 node->hj_OuterCurTuple 中
				if (!ExecScanHashBucket(node, econtext))
				{
					/* out of matches; check for possible outer-join fill */
					// 没有找到匹配的元组，下一步应该从另一张表取出元组，进行哈希
					// 切换探测状态
					node->hj_FetchingFromInner = !node->hj_FetchingFromInner;
					node->hj_JoinState = HJ_GET_AND_HASH_TUPLE;  // 设置节点状态为获取并哈希元组
					continue;
				}

				/*
				 * We've got a match, but still need to test non-hashed quals.
				 * ExecScanHashBucket already set up all the state needed to
				 * call ExecQual.
				 *
				 * If we pass the qual, then save state for next call and have
				 * ExecProject form the projection, store it in the tuple
				 * table, and return the slot.
				 *
				 * Only the joinquals determine tuple match status, but all
				 * quals must pass to actually return the tuple.
				 */
				// 如果找到匹配的内部元组，首先测试非哈希条件（即其他过滤条件）
				if (joinqual == NULL || ExecQual(joinqual, econtext))
				{
					node->hj_MatchedOuter = true;  // 标记已找到与外部元组匹配的内部元组

					/*
						* This is really only needed if HJ_FILL_INNER(node),
						* but we'll avoid the branch and just set it always.
						*/
					HeapTupleHeaderSetMatch(HJTUPLE_MINTUPLE(node->hj_OuterCurTuple));

					/* In an antijoin, we never return a matched tuple */
					// if (node->js.jointype == JOIN_ANTI)
					// {
					// 	node->hj_JoinState = HJ_NEED_NEW_OUTER;
					// 	continue;
					// }

					/*
					 * If we only need to join to the first matching inner
					 * tuple, then consider returning this one, but after that
					 * continue with next outer tuple.
					 * 如果只需要连接到第一个匹配的内表元组，那么可以考虑返回这个元组，
                     * 但是在此之后可以继续使用下一个外表元组。
					 */
					// if (node->js.single_match)
					// 	node->hj_JoinState = HJ_NEED_NEW_OUTER;

					// econtext->ecxt_innertuple = node->hj_NullInnerTupleSlot;

					if (otherqual == NULL || ExecQual(otherqual, econtext)) {
						// 注意：由于之前 ExecScanHashBucket 要求待检测元组存放在 ecxt_outertuple
						// 而查询到的元组存放在 ecxt_innertuple
						// 当 hj_FetchingFromInner == true 时，这会导致投影后左右表内容颠倒
						// 因此要交换 ecxt_outertuple 和 ecxt_innertuple
						if (node->hj_FetchingFromInner) {
							TupleTableSlot *tmp = econtext->ecxt_innertuple;
							econtext->ecxt_innertuple = econtext->ecxt_outertuple;
							econtext->ecxt_outertuple = tmp;
						}

						TupleTableSlot *result = ExecProject(node->js.ps.ps_ProjInfo);

						// 匹配成功，记得依然要切换 node->hj_FetchingFromInner
						node->hj_FetchingFromInner = !node->hj_FetchingFromInner;
						// 还要切换状态
						node->hj_JoinState = HJ_GET_AND_HASH_TUPLE;

						return result;  //执行投影操作
					}
					else
						InstrCountFiltered2(node, 1);  //其他条件不匹配
				}
				else
					InstrCountFiltered1(node, 1);  //连接条件不匹配
				break;

			case HJ_FILL_OUTER_TUPLE:

				/*
				 * The current outer tuple has run out of matches, so check
				 * whether to emit a dummy outer-join tuple.  Whether we emit
				 * one or not, the next state is NEED_NEW_OUTER.
				 * 当前外部元组已耗尽匹配项，因此检查是否发出一个虚拟的外连接元组。
                 * 不管是否发出一个，下一个状态是NEED_NEW_OUTER
				 */
				node->hj_JoinState = HJ_NEED_NEW_OUTER;

				// 如果外部元组没有匹配过任何内部元组且当前的连接策略要求填充外部元组
				if (!node->hj_MatchedOuter &&
					HJ_FILL_OUTER(node))
				{
					/*
					 * Generate a fake join tuple with nulls for the inner
					 * tuple, and return it if it passes the non-join quals.
					 * 生成一个假的连接元组，内部元组部分为NULL，
			         * 如果这个元组通过了非连接条件的测试，则返回它
					 */
					// 将执行上下文中的内部元组槽设置为预定义的空元组槽
					// econtext->ecxt_innertuple = node->hj_NullInnerTupleSlot;

					// 如果没有其他非连接条件或者其他条件测试通过
					if (otherqual == NULL || ExecQual(otherqual, econtext)) {
						TupleTableSlot *result = ExecProject(node->js.ps.ps_ProjInfo);
						return result;  // 执行投影操作，生成最终的结果元组
					}
					else
						InstrCountFiltered2(node, 1);  // 如果条件测试未通过，增加过滤统计计数
				}
				break;

			case HJ_FILL_INNER_TUPLES:  // 处理状态为填充内部元组的情况

				/*
				 * We have finished a batch, but we are doing right/full join,
				 * so any unmatched inner tuples in the hashtable have to be
				 * emitted before we continue to the next batch.
				 * 我们已经完成了一个批次的处理，但是由于我们正在执行右连接/全连接/右反连接，
			     * 因此必须在继续下一个批次之前发出哈希表中所有未匹配的内部元组
				 */
				if (!ExecScanHashTableForUnmatched(node, econtext))
				{
					/* no more unmatched tuples */
					// 没有更多的未匹配内部元组
					node->hj_JoinState = HJ_NEED_NEW_BATCH;  // 设置节点状态为“需要新的批次”
					continue;
				}

				/*
				 * Generate a fake join tuple with nulls for the outer tuple,
				 * and return it if it passes the non-join quals.
				 * 生成一个假的连接元组，外部元组部分为NULL，
			     * 如果这个元组通过了非连接条件的测试，则返回它
				 */
				// 将执行上下文中的外部元组槽设置为预定义的空元组槽
				// econtext->ecxt_outertuple = node->hj_NullOuterTupleSlot;

				// 如果没有其他非连接条件或者其他条件测试通过
				if (otherqual == NULL || ExecQual(otherqual, econtext)) {
					TupleTableSlot *result = ExecProject(node->js.ps.ps_ProjInfo);
					return result;  // 执行投影操作，生成最终的结果元组
				}
				else
					InstrCountFiltered2(node, 1);  // 如果条件测试未通过，增加过滤统计计数
				break;  // 结束当前case处理

			case HJ_NEED_NEW_BATCH:

				/*
				 * Try to advance to next batch.  Done if there are no more.
				 */
				if (parallel)
				{
					if (!ExecParallelHashJoinNewBatch(node))
						return NULL;	/* end of parallel-aware join */
				}
				else
				{
					if (!ExecHashJoinNewBatch(node))
						return NULL;	/* end of parallel-oblivious join */
				}
				node->hj_JoinState = HJ_NEED_NEW_OUTER;
				break;

			default:
				elog(ERROR, "unrecognized hashjoin state: %d",
					 (int) node->hj_JoinState);
		}
	}
}

/* ----------------------------------------------------------------
 *		ExecHashJoin
 *
 *		Parallel-oblivious version.
 * ----------------------------------------------------------------
 */
static TupleTableSlot *			/* return: a tuple or NULL */
ExecHashJoin(PlanState *pstate)
{
	/*
	 * On sufficiently smart compilers this should be inlined with the
	 * parallel-aware branches removed.
	 */
	return ExecHashJoinImpl(pstate, false);
}

/* ----------------------------------------------------------------
 *		ExecParallelHashJoin
 *
 *		Parallel-aware version.
 * ----------------------------------------------------------------
 */
static TupleTableSlot *			/* return: a tuple or NULL */
ExecParallelHashJoin(PlanState *pstate)
{
	/*
	 * On sufficiently smart compilers this should be inlined with the
	 * parallel-oblivious branches removed.
	 */
	return ExecHashJoinImpl(pstate, true);
}

/* ----------------------------------------------------------------
 *		ExecInitHashJoin
 *
 *		Init routine for HashJoin node.
 * ----------------------------------------------------------------
 */
// 定义一个初始化哈希连接节点的函数，该函数返回一个指向哈希连接状态结构的指针
HashJoinState *
ExecInitHashJoin(HashJoin *node, EState *estate, int eflags)
{
	HashJoinState *hjstate;
	// Plan	   *outerNode;
	// Hash	   *hashNode;
	// 新增的哈希节点
	Hash	   *innerHashNode;
	Hash	   *outerHashNode;
	TupleDesc	outerDesc,
				innerDesc;
	const TupleTableSlotOps *ops;
	List	   *lclauses;
	List	   *rclauses;
	List	   *hoperators;
	ListCell   *l;

	/* check for unsupported flags */
	Assert(!(eflags & (EXEC_FLAG_BACKWARD | EXEC_FLAG_MARK)));

	/*
	 * create state structure
	 */
	hjstate = makeNode(HashJoinState);
	hjstate->js.ps.plan = (Plan *) node;
	hjstate->js.ps.state = estate;

	/*
	 * See ExecHashJoinInitializeDSM() and ExecHashJoinInitializeWorker()
	 * where this function may be replaced with a parallel version, if we
	 * managed to launch a parallel query.
	 */
	hjstate->js.ps.ExecProcNode = ExecHashJoin;
	hjstate->js.jointype = node->join.jointype;

	/*
	 * Miscellaneous initialization
	 *
	 * create expression context for node
	 */
	ExecAssignExprContext(estate, &hjstate->js.ps);

	/*
	 * initialize child nodes
	 *
	 * Note: we could suppress the REWIND flag for the inner input, which
	 * would amount to betting that the hash will be a single batch.  Not
	 * clear if this would be a win or not.
	 */
	// 类型转换内外节点
	innerHashNode = (Hash *) innerPlan(node);
	outerHashNode = (Hash *) outerPlan(node);

	// 初始化内外节点
	outerPlanState(hjstate) = ExecInitNode((Plan *) outerHashNode, estate, eflags);
	outerDesc = ExecGetResultType(outerPlanState(hjstate));
	innerPlanState(hjstate) = ExecInitNode((Plan *) innerHashNode, estate, eflags);
	innerDesc = ExecGetResultType(innerPlanState(hjstate));

	/*
	 * Initialize result slot, type and projection.
	 */
	ExecInitResultTupleSlotTL(&hjstate->js.ps, &TTSOpsVirtual);
	ExecAssignProjectionInfo(&hjstate->js.ps, NULL);

	/*
	 * tuple table initialization
	 */
	ops = ExecGetResultSlotOps(outerPlanState(hjstate), NULL);
	hjstate->hj_OuterTupleSlot = ExecInitExtraTupleSlot(estate, outerDesc,
														ops);
	// 内表元组槽也要初始化
	hjstate->hj_InnerTupleSlot = ExecInitExtraTupleSlot(estate, innerDesc,
														ops);

	/*
	 * detect whether we need only consider the first matching inner tuple
	 */
	hjstate->js.single_match = (node->join.inner_unique ||
								node->join.jointype == JOIN_SEMI);

	// 固定使用 JOIN_LEFT
	// 对于实验2.1，不要固定使用 JOIN_LEFT
	// node->join.jointype = JOIN_LEFT;

	// 根据连接类型初始化空元组槽
	switch (node->join.jointype)
	{
		// 对于内连接、半连接等，不需要空元组槽
		case JOIN_INNER:
		case JOIN_SEMI:
			break;
		// 对于左连接、反连接等，初始化内部空元组槽
		case JOIN_LEFT:
		case JOIN_ANTI:
			hjstate->hj_NullInnerTupleSlot =
				ExecInitNullTupleSlot(estate, innerDesc, &TTSOpsVirtual);
			break;
		// 对于右连接，初始化外部空元组槽
		case JOIN_RIGHT:
			hjstate->hj_NullOuterTupleSlot =
				ExecInitNullTupleSlot(estate, outerDesc, &TTSOpsVirtual);
			break;
		// 对于全连接，初始化内外部空元组槽
		case JOIN_FULL:
			hjstate->hj_NullOuterTupleSlot =
				ExecInitNullTupleSlot(estate, outerDesc, &TTSOpsVirtual);
			hjstate->hj_NullInnerTupleSlot =
				ExecInitNullTupleSlot(estate, innerDesc, &TTSOpsVirtual);
			break;
		default:
			elog(ERROR, "unrecognized join type: %d",
				 (int) node->join.jointype);
	}

	/*
	 * now for some voodoo.  our temporary tuple slot is actually the result
	 * tuple slot of the Hash node (which is our inner plan).  we can do this
	 * because Hash nodes don't return tuples via ExecProcNode() -- instead
	 * the hash join node uses ExecScanHashBucket() to get at the contents of
	 * the hash table.  -cim 6/9/91
	 */
	// 将哈希节点的结果元组槽指针赋值给哈希连接的哈希元组槽，用于存储哈希表的元组
	{
		HashState  *hashstate = (HashState *) innerPlanState(hjstate);
		TupleTableSlot *slot = hashstate->ps.ps_ResultTupleSlot;
		hjstate->hj_InnerHashTupleSlot = slot;

		// 外表也要做相同的事情
		hashstate = (HashState *) outerPlanState(hjstate);
		slot = hashstate->ps.ps_ResultTupleSlot;
		hjstate->hj_OuterHashTupleSlot = slot;
	}

	/*
	 * initialize child expressions
	 */
	hjstate->js.ps.qual =
		ExecInitQual(node->join.plan.qual, (PlanState *) hjstate);
	hjstate->js.jointype = node->join.jointype;
	hjstate->js.joinqual =
		ExecInitQual(node->join.joinqual, (PlanState *) hjstate);
	hjstate->hashclauses =
		ExecInitQual(node->hashclauses, (PlanState *) hjstate);

	/*
	 * initialize hash-specific info
	 */
	// 内外哈希表赋 NULL
	hjstate->hj_OuterHashTable = NULL;
	hjstate->hj_InnerHashTable = NULL;

	// 内外表的首位元组槽赋 NULL
	hjstate->hj_FirstOuterTupleSlot = NULL;
	hjstate->hj_FirstInnerTupleSlot = NULL;

	// 内外表的当前哈希值
	hjstate->hj_OuterCurHashValue = 0;
	hjstate->hj_InnerCurHashValue = 0;

	// 哈希桶初始为0
	hjstate->hj_OuterCurBucketNo = 0;
	// 不需要使用 skew table
	hjstate->hj_OuterCurSkewBucketNo = INVALID_SKEW_BUCKET_NO;

	// 内外表当前元组赋 NULL
	hjstate->hj_OuterCurTuple = NULL;
	hjstate->hj_InnerCurTuple = NULL;

	hjstate->hj_OuterHashKeys = ExecInitExprList(node->hashkeys,
												 (PlanState *) hjstate);
	hjstate->hj_InnerHashKeys = ExecInitExprList(node->hashkeys,
												 (PlanState *) hjstate);
	hjstate->hj_HashOperators = node->hashoperators;
	hjstate->hj_Collations = node->hashcollations;

	hjstate->hj_JoinState = HJ_BUILD_HASHTABLE;
	hjstate->hj_MatchedOuter = false;

	// 内外表为空
	hjstate->hj_OuterNotEmpty = false;
	hjstate->hj_InnerNotEmpty = false;

	hjstate->hj_NeedNewInnerTuple = true;
	hjstate->hj_NeedNewOuterTuple = true;

	hjstate->hj_InnerEnd = false;
	hjstate->hj_OuterEnd = false;

	hjstate->hj_FoundByProbingInner = 0;
	hjstate->hj_FoundByProbingOuter = 0;

	// 一开始要从内部关系中读取一个元组
	hjstate->hj_FetchingFromInner = true;

	/*
	 * Deconstruct the hash clauses into outer and inner argument values, so
	 * that we can evaluate those subexpressions separately.  Also make a list
	 * of the hash operator OIDs, in preparation for looking up the hash
	 * functions to use.
	 */
	lclauses = NIL;
	rclauses = NIL;
	hoperators = NIL;
	foreach(l, node->hashclauses)
	{
		OpExpr	   *hclause = lfirst_node(OpExpr, l);
		lclauses = lappend(lclauses, ExecInitExpr(linitial(hclause->args),
												  (PlanState *) hjstate));
		rclauses = lappend(rclauses, ExecInitExpr(lsecond(hclause->args),
												  (PlanState *) hjstate));
		hoperators = lappend_oid(hoperators, hclause->opno);
	}
	hjstate->hj_OuterHashKeys = lclauses;
	hjstate->hj_InnerHashKeys = rclauses;
	hjstate->hj_HashOperators = hoperators;
	/* child Hash node needs to evaluate inner hash keys, too */
	((HashState *) innerPlanState(hjstate))->hashkeys = rclauses;
	// 新增
	((HashState *) outerPlanState(hjstate))->hashkeys = lclauses;

	hjstate->hj_JoinState = HJ_BUILD_HASHTABLE;
	hjstate->hj_MatchedOuter = false;
	hjstate->hj_OuterNotEmpty = false;

	return hjstate;
}

/* ----------------------------------------------------------------
 *		ExecEndHashJoin
 *
 *		clean up routine for HashJoin node
 * ----------------------------------------------------------------
 */
void
ExecEndHashJoin(HashJoinState *node)
{
	/*
	 * Free hash table
	 */
	if (node->hj_OuterHashTable)
	{
		ExecHashTableDestroy(node->hj_OuterHashTable);
		node->hj_OuterHashTable = NULL;
	}

	// 内表也要清理
	if (node->hj_InnerHashTable)
	{
		ExecHashTableDestroy(node->hj_InnerHashTable);
		node->hj_InnerHashTable = NULL;
	}

	// 清理 econtext 的内外表元组槽
	// if (node->js.ps.ps_ExprContext->ecxt_innertuple) {
	// 	ExecClearTuple(node->js.ps.ps_ExprContext->ecxt_innertuple);
	// }
	// if (node->js.ps.ps_ExprContext->ecxt_outertuple) {
	// 	ExecClearTuple(node->js.ps.ps_ExprContext->ecxt_outertuple);
	// }

	// 清理 TupleDesc
	// if (node->hj_InnerTupleSlot->tts_tupleDescriptor) {
	// 	FreeTupleDesc(node->hj_InnerTupleSlot->tts_tupleDescriptor);
	// }
	// if (node->hj_OuterTupleSlot->tts_tupleDescriptor) {
	// 	FreeTupleDesc(node->hj_OuterTupleSlot->tts_tupleDescriptor);
	// }

	/*
	 * Free the exprcontext
	 */
	ExecFreeExprContext(&node->js.ps);

	/*
	 * clean out the tuple table
	 */
	ExecClearTuple(node->js.ps.ps_ResultTupleSlot);
	if (node->hj_OuterTupleSlot) {
		ExecClearTuple(node->hj_OuterTupleSlot);
	}
	if (node->hj_InnerTupleSlot) {
		ExecClearTuple(node->hj_InnerTupleSlot);
	}
	ExecClearTuple(node->hj_OuterHashTupleSlot);

	/*
	 * clean up subtrees
	 */
	ExecEndNode(outerPlanState(node));
	ExecEndNode(innerPlanState(node));
}

// 自己写的获取元组
// 会直接通过 hashNode 的 SeqScan 节点获取下一个元组
// 用于当一张表耗尽后，另一张表的扫描
static TupleTableSlot *
ExecGetTuple(HashState *hashNode,
			 HashJoinState *hjstate
)
{
	PlanState *ssNode;
	HashJoinTable hashtable;
	List *hashkeys;
	ExprContext *econtext;
	TupleTableSlot *slot;

	ssNode = hashNode->ps.lefttree;
	hashtable = hashNode->hashtable;
	hashkeys = hashNode->hashkeys;
	econtext = hashNode->ps.ps_ExprContext;

	/*
	* Check to see if first outer tuple was already fetched by
	* ExecHashJoin() and not used yet.
	*/
	if (hjstate->hj_FetchingFromInner) {
		slot = hjstate->hj_FirstInnerTupleSlot;
	} else {
		slot = hjstate->hj_FirstOuterTupleSlot;
	}

	if (!TupIsNull(slot)) {
		if (hjstate->hj_FetchingFromInner) {
			hjstate->hj_FirstInnerTupleSlot = NULL;
		} else {
			hjstate->hj_FirstOuterTupleSlot = NULL;
		}
	}
		// hjstate->hj_FirstOuterTupleSlot = NULL;

	else
		// 从 SeqScanNode 取元组
		slot = ExecProcNode(ssNode);

	if (TupIsNull(slot)) {
		return NULL;
	}
	return slot;
}

/*
 * ExecHashJoinOuterGetTuple
 *
 *		get the next outer tuple for a parallel oblivious hashjoin: either by
 *		executing the outer plan node in the first pass, or from the temp
 *		files for the hashjoin batches.
 *
 * Returns a null slot if no more outer tuples (within the current batch).
 *
 * On success, the tuple's hash value is stored at *outerHashvalue --- this is
 * either originally computed, or re-read from the temp file.
 */
static TupleTableSlot *
ExecHashJoinOuterGetTuple(PlanState *outerNode,
						  HashJoinState *hjstate,
						  uint32 *outerHashvalue)
{
	// 这里要根据 hj_FoundByProbingInner 来选择
	HashJoinTable hashtable;
	List *hashkeys;
	if (hjstate->hj_FoundByProbingInner) {
		hashtable = hjstate->hj_InnerHashTable;
		hashkeys = hjstate->hj_InnerHashKeys;
	} else {
		hashtable = hjstate->hj_OuterHashTable;
		hashkeys = hjstate->hj_OuterHashKeys;
	}
	hashtable = ((HashState *) outerNode)->hashtable;
	hashkeys = ((HashState *) outerNode)->hashkeys;
	ExprContext *econtext = ((HashState *) outerNode)->ps.ps_ExprContext;
	int			curbatch = hashtable->curbatch;
	TupleTableSlot *slot;

	if (curbatch == 0)			/* if it is the first pass */
	{
		/*
		 * Check to see if first outer tuple was already fetched by
		 * ExecHashJoin() and not used yet.
		 */
		slot = hjstate->hj_FirstOuterTupleSlot;
		if (!TupIsNull(slot))
			hjstate->hj_FirstOuterTupleSlot = NULL;
		else
			slot = ExecProcNode(outerNode);

		while (!TupIsNull(slot))
		{
			/*
			 * We have to compute the tuple's hash value.
			 */
			// ExprContext *econtext = hjstate->js.ps.ps_ExprContext;

			econtext->ecxt_outertuple = slot;
			if (ExecHashGetHashValue(hashtable, econtext,
									 hashkeys,
									 true,	/* outer tuple */
									 HJ_FILL_OUTER(hjstate),
									 outerHashvalue))
			{
				/* remember outer relation is not empty for possible rescan */
				hjstate->hj_OuterNotEmpty = true;

				return slot;
			}

			/*
			 * That tuple couldn't match because of a NULL, so discard it and
			 * continue with the next one.
			 */
			slot = ExecProcNode(outerNode);
		}
	}
	else if (curbatch < hashtable->nbatch)
	{
		BufFile    *file = hashtable->outerBatchFile[curbatch];

		/*
		 * In outer-join cases, we could get here even though the batch file
		 * is empty.
		 */
		if (file == NULL)
			return NULL;

		slot = ExecHashJoinGetSavedTuple(hjstate,
										 file,
										 outerHashvalue,
										 hjstate->hj_OuterTupleSlot);
		if (!TupIsNull(slot))
			return slot;
	}

	/* End of this batch */
	return NULL;
}

/*
 * ExecHashJoinOuterGetTuple variant for the parallel case.
 */
static TupleTableSlot *
ExecParallelHashJoinOuterGetTuple(PlanState *outerNode,
								  HashJoinState *hjstate,
								  uint32 *outerHashvalue)
{
	HashJoinTable hashtable = hjstate->hj_OuterHashTable;
	int			curbatch = hashtable->curbatch;
	TupleTableSlot *slot;

	/*
	 * In the Parallel Hash case we only run the outer plan directly for
	 * single-batch hash joins.  Otherwise we have to go to batch files, even
	 * for batch 0.
	 */
	if (curbatch == 0 && hashtable->nbatch == 1)
	{
		slot = ExecProcNode(outerNode);

		while (!TupIsNull(slot))
		{
			ExprContext *econtext = hjstate->js.ps.ps_ExprContext;

			econtext->ecxt_outertuple = slot;
			if (ExecHashGetHashValue(hashtable, econtext,
									 hjstate->hj_OuterHashKeys,
									 true,	/* outer tuple */
									 HJ_FILL_OUTER(hjstate),
									 outerHashvalue))
				return slot;

			/*
			 * That tuple couldn't match because of a NULL, so discard it and
			 * continue with the next one.
			 */
			slot = ExecProcNode(outerNode);
		}
	}
	else if (curbatch < hashtable->nbatch)
	{
		MinimalTuple tuple;

		tuple = sts_parallel_scan_next(hashtable->batches[curbatch].outer_tuples,
									   outerHashvalue);
		if (tuple != NULL)
		{
			ExecForceStoreMinimalTuple(tuple,
									   hjstate->hj_OuterTupleSlot,
									   false);
			slot = hjstate->hj_OuterTupleSlot;
			return slot;
		}
		else
			ExecClearTuple(hjstate->hj_OuterTupleSlot);
	}

	/* End of this batch */
	return NULL;
}

/*
 * ExecHashJoinNewBatch
 *		switch to a new hashjoin batch
 *
 * Returns true if successful, false if there are no more batches.
 */
static bool
ExecHashJoinNewBatch(HashJoinState *hjstate)
{
	HashJoinTable hashtable = hjstate->hj_OuterHashTable;
	int			nbatch;
	int			curbatch;
	BufFile    *innerFile;
	TupleTableSlot *slot;
	uint32		outerHashvalue;

	nbatch = hashtable->nbatch;
	curbatch = hashtable->curbatch;

	if (curbatch > 0)
	{
		/*
		 * We no longer need the previous outer batch file; close it right
		 * away to free disk space.
		 */
		if (hashtable->outerBatchFile[curbatch])
			BufFileClose(hashtable->outerBatchFile[curbatch]);
		hashtable->outerBatchFile[curbatch] = NULL;
	}
	else						/* we just finished the first batch */
	{
		/*
		 * Reset some of the skew optimization state variables, since we no
		 * longer need to consider skew tuples after the first batch. The
		 * memory context reset we are about to do will release the skew
		 * hashtable itself.
		 */
		hashtable->skewEnabled = false;
		hashtable->skewBucket = NULL;
		hashtable->skewBucketNums = NULL;
		hashtable->nSkewBuckets = 0;
		hashtable->spaceUsedSkew = 0;
	}

	/*
	 * We can always skip over any batches that are completely empty on both
	 * sides.  We can sometimes skip over batches that are empty on only one
	 * side, but there are exceptions:
	 *
	 * 1. In a left/full outer join, we have to process outer batches even if
	 * the inner batch is empty.  Similarly, in a right/full outer join, we
	 * have to process inner batches even if the outer batch is empty.
	 *
	 * 2. If we have increased nbatch since the initial estimate, we have to
	 * scan inner batches since they might contain tuples that need to be
	 * reassigned to later inner batches.
	 *
	 * 3. Similarly, if we have increased nbatch since starting the outer
	 * scan, we have to rescan outer batches in case they contain tuples that
	 * need to be reassigned.
	 */
	curbatch++;
	while (curbatch < nbatch &&
		   (hashtable->outerBatchFile[curbatch] == NULL ||
			hashtable->innerBatchFile[curbatch] == NULL))
	{
		if (hashtable->outerBatchFile[curbatch] &&
			HJ_FILL_OUTER(hjstate))
			break;				/* must process due to rule 1 */
		if (hashtable->innerBatchFile[curbatch] &&
			HJ_FILL_INNER(hjstate))
			break;				/* must process due to rule 1 */
		if (hashtable->innerBatchFile[curbatch] &&
			nbatch != hashtable->nbatch_original)
			break;				/* must process due to rule 2 */
		if (hashtable->outerBatchFile[curbatch] &&
			nbatch != hashtable->nbatch_outstart)
			break;				/* must process due to rule 3 */
		/* We can ignore this batch. */
		/* Release associated temp files right away. */
		if (hashtable->innerBatchFile[curbatch])
			BufFileClose(hashtable->innerBatchFile[curbatch]);
		hashtable->innerBatchFile[curbatch] = NULL;
		if (hashtable->outerBatchFile[curbatch])
			BufFileClose(hashtable->outerBatchFile[curbatch]);
		hashtable->outerBatchFile[curbatch] = NULL;
		curbatch++;
	}

	if (curbatch >= nbatch)
		return false;			/* no more batches */

	hashtable->curbatch = curbatch;

	/*
	 * Reload the hash table with the new inner batch (which could be empty)
	 */
	ExecHashTableReset(hashtable);

	innerFile = hashtable->innerBatchFile[curbatch];

	if (innerFile != NULL)
	{
		if (BufFileSeek(innerFile, 0, 0L, SEEK_SET))
			ereport(ERROR,
					(errcode_for_file_access(),
					 errmsg("could not rewind hash-join temporary file")));

		while ((slot = ExecHashJoinGetSavedTuple(hjstate,
												 innerFile,
												 &outerHashvalue,
												 hjstate->hj_OuterHashTupleSlot)))
		{
			/*
			 * NOTE: some tuples may be sent to future batches.  Also, it is
			 * possible for hashtable->nbatch to be increased here!
			 */
			ExecHashTableInsert(hashtable, slot, outerHashvalue);
		}

		/*
		 * after we build the hash table, the inner batch file is no longer
		 * needed
		 */
		BufFileClose(innerFile);
		hashtable->innerBatchFile[curbatch] = NULL;
	}

	/*
	 * Rewind outer batch file (if present), so that we can start reading it.
	 */
	if (hashtable->outerBatchFile[curbatch] != NULL)
	{
		if (BufFileSeek(hashtable->outerBatchFile[curbatch], 0, 0L, SEEK_SET))
			ereport(ERROR,
					(errcode_for_file_access(),
					 errmsg("could not rewind hash-join temporary file")));
	}

	return true;
}

/*
 * Choose a batch to work on, and attach to it.  Returns true if successful,
 * false if there are no more batches.
 */
static bool
ExecParallelHashJoinNewBatch(HashJoinState *hjstate)
{
	HashJoinTable hashtable = hjstate->hj_OuterHashTable;
	int			start_batchno;
	int			batchno;

	/*
	 * If we started up so late that the batch tracking array has been freed
	 * already by ExecHashTableDetach(), then we are finished.  See also
	 * ExecParallelHashEnsureBatchAccessors().
	 */
	if (hashtable->batches == NULL)
		return false;

	/*
	 * If we were already attached to a batch, remember not to bother checking
	 * it again, and detach from it (possibly freeing the hash table if we are
	 * last to detach).
	 */
	if (hashtable->curbatch >= 0)
	{
		hashtable->batches[hashtable->curbatch].done = true;
		ExecHashTableDetachBatch(hashtable);
	}

	/*
	 * Search for a batch that isn't done.  We use an atomic counter to start
	 * our search at a different batch in every participant when there are
	 * more batches than participants.
	 */
	batchno = start_batchno =
		pg_atomic_fetch_add_u32(&hashtable->parallel_state->distributor, 1) %
		hashtable->nbatch;
	do
	{
		uint32		outerHashvalue;
		MinimalTuple tuple;
		TupleTableSlot *slot;

		if (!hashtable->batches[batchno].done)
		{
			SharedTuplestoreAccessor *inner_tuples;
			Barrier    *batch_barrier =
			&hashtable->batches[batchno].shared->batch_barrier;

			switch (BarrierAttach(batch_barrier))
			{
				case PHJ_BATCH_ELECTING:

					/* One backend allocates the hash table. */
					if (BarrierArriveAndWait(batch_barrier,
											 WAIT_EVENT_HASH_BATCH_ELECTING))
						ExecParallelHashTableAlloc(hashtable, batchno);
					/* Fall through. */

				case PHJ_BATCH_ALLOCATING:
					/* Wait for allocation to complete. */
					BarrierArriveAndWait(batch_barrier,
										 WAIT_EVENT_HASH_BATCH_ALLOCATING);
					/* Fall through. */

				case PHJ_BATCH_LOADING:
					/* Start (or join in) loading tuples. */
					ExecParallelHashTableSetCurrentBatch(hashtable, batchno);
					inner_tuples = hashtable->batches[batchno].inner_tuples;
					sts_begin_parallel_scan(inner_tuples);
					while ((tuple = sts_parallel_scan_next(inner_tuples,
														   &outerHashvalue)))
					{
						ExecForceStoreMinimalTuple(tuple,
												   hjstate->hj_OuterHashTupleSlot,
												   false);
						slot = hjstate->hj_OuterHashTupleSlot;
						ExecParallelHashTableInsertCurrentBatch(hashtable, slot,
																outerHashvalue);
					}
					sts_end_parallel_scan(inner_tuples);
					BarrierArriveAndWait(batch_barrier,
										 WAIT_EVENT_HASH_BATCH_LOADING);
					/* Fall through. */

				case PHJ_BATCH_PROBING:

					/*
					 * This batch is ready to probe.  Return control to
					 * caller. We stay attached to batch_barrier so that the
					 * hash table stays alive until everyone's finished
					 * probing it, but no participant is allowed to wait at
					 * this barrier again (or else a deadlock could occur).
					 * All attached participants must eventually call
					 * BarrierArriveAndDetach() so that the final phase
					 * PHJ_BATCH_DONE can be reached.
					 */
					ExecParallelHashTableSetCurrentBatch(hashtable, batchno);
					sts_begin_parallel_scan(hashtable->batches[batchno].outer_tuples);
					return true;

				case PHJ_BATCH_DONE:

					/*
					 * Already done.  Detach and go around again (if any
					 * remain).
					 */
					BarrierDetach(batch_barrier);
					hashtable->batches[batchno].done = true;
					hashtable->curbatch = -1;
					break;

				default:
					elog(ERROR, "unexpected batch phase %d",
						 BarrierPhase(batch_barrier));
			}
		}
		batchno = (batchno + 1) % hashtable->nbatch;
	} while (batchno != start_batchno);

	return false;
}

/*
 * ExecHashJoinSaveTuple
 *		save a tuple to a batch file.
 *
 * The data recorded in the file for each tuple is its hash value,
 * then the tuple in MinimalTuple format.
 *
 * Note: it is important always to call this in the regular executor
 * context, not in a shorter-lived context; else the temp file buffers
 * will get messed up.
 */
void
ExecHashJoinSaveTuple(MinimalTuple tuple, uint32 outerHashvalue,
					  BufFile **fileptr)
{
	BufFile    *file = *fileptr;

	if (file == NULL)
	{
		/* First write to this batch file, so open it. */
		file = BufFileCreateTemp(false);
		*fileptr = file;
	}

	BufFileWrite(file, (void *) &outerHashvalue, sizeof(uint32));
	BufFileWrite(file, (void *) tuple, tuple->t_len);
}

/*
 * ExecHashJoinGetSavedTuple
 *		read the next tuple from a batch file.  Return NULL if no more.
 *
 * On success, *outerHashvalue is set to the tuple's hash value, and the tuple
 * itself is stored in the given slot.
 */
static TupleTableSlot *
ExecHashJoinGetSavedTuple(HashJoinState *hjstate,
						  BufFile *file,
						  uint32 *outerHashvalue,
						  TupleTableSlot *tupleSlot)
{
	uint32		header[2];
	size_t		nread;
	MinimalTuple tuple;

	/*
	 * We check for interrupts here because this is typically taken as an
	 * alternative code path to an ExecProcNode() call, which would include
	 * such a check.
	 */
	CHECK_FOR_INTERRUPTS();

	/*
	 * Since both the hash value and the MinimalTuple length word are uint32,
	 * we can read them both in one BufFileRead() call without any type
	 * cheating.
	 */
	nread = BufFileRead(file, (void *) header, sizeof(header));
	if (nread == 0)				/* end of file */
	{
		ExecClearTuple(tupleSlot);
		return NULL;
	}
	if (nread != sizeof(header))
		ereport(ERROR,
				(errcode_for_file_access(),
				 errmsg("could not read from hash-join temporary file: read only %zu of %zu bytes",
						nread, sizeof(header))));
	*outerHashvalue = header[0];
	tuple = (MinimalTuple) palloc(header[1]);
	tuple->t_len = header[1];
	nread = BufFileRead(file,
						(void *) ((char *) tuple + sizeof(uint32)),
						header[1] - sizeof(uint32));
	if (nread != header[1] - sizeof(uint32))
		ereport(ERROR,
				(errcode_for_file_access(),
				 errmsg("could not read from hash-join temporary file: read only %zu of %zu bytes",
						nread, header[1] - sizeof(uint32))));
	ExecForceStoreMinimalTuple(tuple, tupleSlot, true);
	return tupleSlot;
}


void
ExecReScanHashJoin(HashJoinState *node)
{
	/*
	 * In a multi-batch join, we currently have to do rescans the hard way,
	 * primarily because batch temp files may have already been released. But
	 * if it's a single-batch join, and there is no parameter change for the
	 * inner subnode, then we can just re-use the existing hash table without
	 * rebuilding it.
	 */
	if (node->hj_OuterHashTable != NULL)
	{
		if (node->hj_OuterHashTable->nbatch == 1 &&
			node->js.ps.righttree->chgParam == NULL)
		{
			/*
			 * Okay to reuse the hash table; needn't rescan inner, either.
			 *
			 * However, if it's a right/full join, we'd better reset the
			 * inner-tuple match flags contained in the table.
			 */
			if (HJ_FILL_INNER(node))
				ExecHashTableResetMatchFlags(node->hj_OuterHashTable);

			/*
			 * Also, we need to reset our state about the emptiness of the
			 * outer relation, so that the new scan of the outer will update
			 * it correctly if it turns out to be empty this time. (There's no
			 * harm in clearing it now because ExecHashJoin won't need the
			 * info.  In the other cases, where the hash table doesn't exist
			 * or we are destroying it, we leave this state alone because
			 * ExecHashJoin will need it the first time through.)
			 */
			node->hj_OuterNotEmpty = false;

			/* ExecHashJoin can skip the BUILD_HASHTABLE step */
			node->hj_JoinState = HJ_NEED_NEW_OUTER;
		}
		else
		{
			/* must destroy and rebuild hash table */
			HashState  *hashNode = castNode(HashState, innerPlanState(node));

			/* for safety, be sure to clear child plan node's pointer too */
			Assert(hashNode->hashtable == node->hj_OuterHashTable);
			hashNode->hashtable = NULL;

			ExecHashTableDestroy(node->hj_OuterHashTable);
			node->hj_OuterHashTable = NULL;
			node->hj_JoinState = HJ_BUILD_HASHTABLE;

			/*
			 * if chgParam of subnode is not null then plan will be re-scanned
			 * by first ExecProcNode.
			 */
			if (node->js.ps.righttree->chgParam == NULL)
				ExecReScan(node->js.ps.righttree);
		}
	}

	/* Always reset intra-tuple state */
	node->hj_OuterCurHashValue = 0;
	node->hj_OuterCurBucketNo = 0;
	node->hj_OuterCurSkewBucketNo = INVALID_SKEW_BUCKET_NO;
	node->hj_OuterCurTuple = NULL;

	node->hj_MatchedOuter = false;
	node->hj_FirstOuterTupleSlot = NULL;

	/*
	 * if chgParam of subnode is not null then plan will be re-scanned by
	 * first ExecProcNode.
	 */
	if (node->js.ps.lefttree->chgParam == NULL)
		ExecReScan(node->js.ps.lefttree);
}

void
ExecShutdownHashJoin(HashJoinState *node)
{
	if (node->hj_OuterHashTable)
	{
		/*
		 * Detach from shared state before DSM memory goes away.  This makes
		 * sure that we don't have any pointers into DSM memory by the time
		 * ExecEndHashJoin runs.
		 */
		ExecHashTableDetachBatch(node->hj_OuterHashTable);
		ExecHashTableDetach(node->hj_OuterHashTable);
	}
}

static void
ExecParallelHashJoinPartitionOuter(HashJoinState *hjstate)
{
	PlanState  *outerState = outerPlanState(hjstate);
	ExprContext *econtext = hjstate->js.ps.ps_ExprContext;
	HashJoinTable hashtable = hjstate->hj_OuterHashTable;
	TupleTableSlot *slot;
	uint32		outerHashvalue;
	int			i;

	Assert(hjstate->hj_FirstOuterTupleSlot == NULL);

	/* Execute outer plan, writing all tuples to shared tuplestores. */
	for (;;)
	{
		slot = ExecProcNode(outerState);
		if (TupIsNull(slot))
			break;
		econtext->ecxt_outertuple = slot;
		if (ExecHashGetHashValue(hashtable, econtext,
								 hjstate->hj_OuterHashKeys,
								 true,	/* outer tuple */
								 HJ_FILL_OUTER(hjstate),
								 &outerHashvalue))
		{
			int			batchno;
			int			bucketno;
			bool		shouldFree;
			MinimalTuple mintup = ExecFetchSlotMinimalTuple(slot, &shouldFree);

			ExecHashGetBucketAndBatch(hashtable, outerHashvalue, &bucketno,
									  &batchno);
			sts_puttuple(hashtable->batches[batchno].outer_tuples,
						 &outerHashvalue, mintup);

			if (shouldFree)
				heap_free_minimal_tuple(mintup);
		}
		CHECK_FOR_INTERRUPTS();
	}

	/* Make sure all outer partitions are readable by any backend. */
	for (i = 0; i < hashtable->nbatch; ++i)
		sts_end_write(hashtable->batches[i].outer_tuples);
}

void
ExecHashJoinEstimate(HashJoinState *state, ParallelContext *pcxt)
{
	shm_toc_estimate_chunk(&pcxt->estimator, sizeof(ParallelHashJoinState));
	shm_toc_estimate_keys(&pcxt->estimator, 1);
}

void
ExecHashJoinInitializeDSM(HashJoinState *state, ParallelContext *pcxt)
{
	int			plan_node_id = state->js.ps.plan->plan_node_id;
	HashState  *hashNode;
	ParallelHashJoinState *pstate;

	/*
	 * Disable shared hash table mode if we failed to create a real DSM
	 * segment, because that means that we don't have a DSA area to work with.
	 */
	if (pcxt->seg == NULL)
		return;

	ExecSetExecProcNode(&state->js.ps, ExecParallelHashJoin);

	/*
	 * Set up the state needed to coordinate access to the shared hash
	 * table(s), using the plan node ID as the toc key.
	 */
	pstate = shm_toc_allocate(pcxt->toc, sizeof(ParallelHashJoinState));
	shm_toc_insert(pcxt->toc, plan_node_id, pstate);

	/*
	 * Set up the shared hash join state with no batches initially.
	 * ExecHashTableCreate() will prepare at least one later and set nbatch
	 * and space_allowed.
	 */
	pstate->nbatch = 0;
	pstate->space_allowed = 0;
	pstate->batches = InvalidDsaPointer;
	pstate->old_batches = InvalidDsaPointer;
	pstate->nbuckets = 0;
	pstate->growth = PHJ_GROWTH_OK;
	pstate->chunk_work_queue = InvalidDsaPointer;
	pg_atomic_init_u32(&pstate->distributor, 0);
	pstate->nparticipants = pcxt->nworkers + 1;
	pstate->total_tuples = 0;
	LWLockInitialize(&pstate->lock,
					 LWTRANCHE_PARALLEL_HASH_JOIN);
	BarrierInit(&pstate->build_barrier, 0);
	BarrierInit(&pstate->grow_batches_barrier, 0);
	BarrierInit(&pstate->grow_buckets_barrier, 0);

	/* Set up the space we'll use for shared temporary files. */
	SharedFileSetInit(&pstate->fileset, pcxt->seg);

	/* Initialize the shared state in the hash node. */
	hashNode = (HashState *) innerPlanState(state);
	hashNode->parallel_state = pstate;
}

/* ----------------------------------------------------------------
 *		ExecHashJoinReInitializeDSM
 *
 *		Reset shared state before beginning a fresh scan.
 * ----------------------------------------------------------------
 */
void
ExecHashJoinReInitializeDSM(HashJoinState *state, ParallelContext *cxt)
{
	int			plan_node_id = state->js.ps.plan->plan_node_id;
	ParallelHashJoinState *pstate =
	shm_toc_lookup(cxt->toc, plan_node_id, false);

	/*
	 * It would be possible to reuse the shared hash table in single-batch
	 * cases by resetting and then fast-forwarding build_barrier to
	 * PHJ_BUILD_DONE and batch 0's batch_barrier to PHJ_BATCH_PROBING, but
	 * currently shared hash tables are already freed by now (by the last
	 * participant to detach from the batch).  We could consider keeping it
	 * around for single-batch joins.  We'd also need to adjust
	 * finalize_plan() so that it doesn't record a dummy dependency for
	 * Parallel Hash nodes, preventing the rescan optimization.  For now we
	 * don't try.
	 */

	/* Detach, freeing any remaining shared memory. */
	if (state->hj_OuterHashTable != NULL)
	{
		ExecHashTableDetachBatch(state->hj_OuterHashTable);
		ExecHashTableDetach(state->hj_OuterHashTable);
	}

	/* Clear any shared batch files. */
	SharedFileSetDeleteAll(&pstate->fileset);

	/* Reset build_barrier to PHJ_BUILD_ELECTING so we can go around again. */
	BarrierInit(&pstate->build_barrier, 0);
}

void
ExecHashJoinInitializeWorker(HashJoinState *state,
							 ParallelWorkerContext *pwcxt)
{
	HashState  *hashNode;
	int			plan_node_id = state->js.ps.plan->plan_node_id;
	ParallelHashJoinState *pstate =
	shm_toc_lookup(pwcxt->toc, plan_node_id, false);

	/* Attach to the space for shared temporary files. */
	SharedFileSetAttach(&pstate->fileset, pwcxt->seg);

	/* Attach to the shared state in the hash node. */
	hashNode = (HashState *) innerPlanState(state);
	hashNode->parallel_state = pstate;

	ExecSetExecProcNode(&state->js.ps, ExecParallelHashJoin);
}
