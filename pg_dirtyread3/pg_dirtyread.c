#include "postgres.h"

#include "access/heapam.h"
#include "access/htup_details.h"
#include "access/table.h"
#include "access/transam.h"
#include "access/xact.h"
#include "access/xlog.h"
#include "access/xlog.h"
#include "catalog/pg_type.h"
#include "fmgr.h"
#include "funcapi.h"
#include "miscadmin.h"
#include "storage/proc.h"
#include "storage/procarray.h"
#include "utils/builtins.h"
#include "utils/datum.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"
#include "utils/snapmgr.h"
#include "utils/syscache.h"

extern int					txn_id;
int					txn_id;


extern void _PG_init(void);
void _PG_init(void)
{
	DefineCustomIntVariable("txn_id", "Sets the dirtyread txn_id",
							"Valid range is 1...maxtxn", &txn_id,
							0, 0, INT_MAX, PGC_USERSET, 0, NULL, NULL, NULL);

}

Datum dirtyread(PG_FUNCTION_ARGS);

typedef struct dirtyread_ctx_state
{
	Relation		rel;
	TupleDesc		desc;
	TableScanDesc	scan;
} dirtyread_ctx_state;

PG_MODULE_MAGIC;

PG_FUNCTION_INFO_V1(dirtyread);
Datum dirtyread(PG_FUNCTION_ARGS)
{
	FuncCallContext *funcctx;
	dirtyread_ctx_state *inter_call_data = NULL;
	HeapTuple tuple;
	TransactionId xmin, xmax; // 遍历表的当前元组的xmin、xmax
	Datum datum;

	if (SRF_IS_FIRSTCALL())
	{
		MemoryContext oldcontext;
		Oid relid;
		TupleDesc tupdesc;

		relid = PG_GETARG_OID(0);
		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		inter_call_data = (dirtyread_ctx_state *)palloc(sizeof(dirtyread_ctx_state));
		inter_call_data->rel = table_open(relid, AccessShareLock);
		inter_call_data->desc = RelationGetDescr(inter_call_data->rel);

		if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
			elog(ERROR, "return type must be a row type");
		funcctx->tuple_desc = BlessTupleDesc(tupdesc);

		inter_call_data->scan = heap_beginscan(inter_call_data->rel, SnapshotAny, 0, NULL, NULL, 0);

		funcctx->user_fctx = (void *)inter_call_data;

		MemoryContextSwitchTo(oldcontext);
	}

	funcctx = SRF_PERCALL_SETUP();
	inter_call_data = (dirtyread_ctx_state *)funcctx->user_fctx;

	// 遍历表中的元组
	while ((tuple = heap_getnext(inter_call_data->scan, ForwardScanDirection)) != NULL)
	{
		xmin = HeapTupleHeaderGetXmin(tuple->t_data);
		xmax = HeapTupleHeaderGetUpdateXid(tuple->t_data);

		// 未设置 txn_id，直接输出所有的快照
		if (txn_id == 0) {
			SRF_RETURN_NEXT(funcctx, heap_copy_tuple_as_datum(tuple, inter_call_data->desc));
		}

		// 如果存在 xmax
		if (!(tuple->t_data->t_infomask & HEAP_XMAX_INVALID)) {
			if (txn_id >= xmax) {
				continue;
			} 
			if (txn_id >= xmin && txn_id < xmax) {
				SRF_RETURN_NEXT(funcctx, heap_copy_tuple_as_datum(tuple, inter_call_data->desc));
			}
		// 如果不存在 xmax
		} else {
			if (txn_id >= xmin) {
				SRF_RETURN_NEXT(funcctx, heap_copy_tuple_as_datum(tuple, inter_call_data->desc));
			}
		}
	}

	// 元组已经遍历完了
	// txn_id = 0;
	heap_endscan(inter_call_data->scan);
	table_close(inter_call_data->rel, AccessShareLock);
	SRF_RETURN_DONE(funcctx);
}