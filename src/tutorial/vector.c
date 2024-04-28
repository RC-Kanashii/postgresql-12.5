/*
 * src/tutorial/vector.c
 *
 ******************************************************************************
  This file contains routines that can be bound to a Postgres backend and
  called by the backend in the process of processing queries.  The calling
  format for these routines is dictated by Postgres architecture.
******************************************************************************/

#include "postgres.h"

#include "fmgr.h"
#include "libpq/pqformat.h"        /* needed for send/recv functions */
#include "common/shortest_dec.h"

#include <string.h>
#include <math.h>

PG_MODULE_MAGIC;

typedef struct Vector {
    int32 vl_len_;  // 永远不会访问
    int32 dim;
    float arr[FLEXIBLE_ARRAY_MEMBER];  // 不要使用指针
} Vector;


/*****************************************************************************
 * Input/Output functions
 *****************************************************************************/

PG_FUNCTION_INFO_V1(vector_in);

Datum
vector_in(PG_FUNCTION_ARGS) {
    char *originalStr = PG_GETARG_CSTRING(0);
    char *token = NULL;
    int32 dim = 0;
    float *arr;  // 维度不会超过1024，因此够用了
    Vector *result;
    char *str;  // mutable的字符串
    char *p;  // 辅助用的字符串指针
    int commaNum;  // 逗号个数
    int leftCurlyBracket;  // 左逗号个数
    int rightCurlyBracket;  // 右逗号个数
    float val;  // 转换的值
    int i;


    // 把字面量变成mutable
    str = strdup(originalStr);

    // 初始化arr
    arr = (float *) palloc0(1030 * sizeof(float));

    // 检查字符串是否以'{'开头，以'}'结尾
    if (str[0] != '{' || str[strlen(str) - 1] != '}') {
        ereport(ERROR,
                (errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
                        errmsg("invalid input syntax for type %s: \"%s\"",
                               "vector", originalStr)));
    }

    // 数逗号的个数
    commaNum = 0;
    leftCurlyBracket = 0, rightCurlyBracket = 0;
    for (p = str; *p != '\0'; p++) {
        if (*p == ',') commaNum++;
        else if (*p == '{') leftCurlyBracket++;
        else if (*p == '}') rightCurlyBracket++;
    }

    if (leftCurlyBracket != 1 || rightCurlyBracket != 1) {
        ereport(ERROR,
                (errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
                        errmsg("invalid input syntax for type %s: \"%s\"",
                               "vector", originalStr)));
    }

    // 跳过开头的'{'
    str++;

    // 使用 strtok 解析每个浮点数
    token = strtok(str, ",}");
    while (token != NULL) {
        // 检查小数点个数，和空格是否合法
        int dotNum = 0;
        int meetDigit = 0;
        for (p = token; *p != '\0'; p++) {
            if (*p == '.') dotNum++;
            else if (isdigit(*p)) meetDigit = 1;
            else if (meetDigit == 1 && *p == ' ' || isalpha(*p)) {
                ereport(ERROR,
                        (errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
                                errmsg("invalid input syntax for type %s: \"%s\"",
                                       "vector", originalStr)));
            }
        }
        if (dotNum > 1 || token[strlen(token) - 1] == '.') {
            ereport(ERROR,
                    (errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
                            errmsg("invalid input syntax for type %s: \"%s\"",
                                   "vector", originalStr)));
        }

        val = strtof(token, NULL);
        // 转换失败的话，会返回0.0f；超过范围的话会返回正负HUGE_VALF
        if (val == 0.0f || val == HUGE_VALF || val == -HUGE_VALF || isnan(val) || isinf(val)) {
            ereport(ERROR,
                    (errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
                            errmsg("invalid input syntax for type %s: \"%s\"",
                                   "vector", originalStr)));
        }
        dim++;
//        arr = repalloc(arr, dim * sizeof(float));  // 这里不要用repalloc
        arr[dim - 1] = val;
        token = strtok(NULL, ",}");  // 指向下一个token
    }

    if (commaNum + 1 != dim) {
        ereport(ERROR,
                (errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
                        errmsg("invalid input syntax for type %s: \"%s\"",
                               "vector", originalStr)));
    }

    // 处理result
    result = (Vector *) palloc0(VARHDRSZ + sizeof(int32) + dim * sizeof(float));
    // result->arr也需要分配内存空间
//    result->arr = (float *) palloc0(dim * sizeof(float));
    // 记得要SET_VARSIZE
    SET_VARSIZE(result, VARHDRSZ + sizeof(int32) + dim * sizeof(float));
    result->dim = dim;
//    memcpy(result->arr, arr, dim * sizeof(float));
    for (i = 0; i < dim; i++) {
        result->arr[i] = arr[i];
    }

    pfree(arr);

    PG_RETURN_POINTER(result);
}

PG_FUNCTION_INFO_V1(vector_out);

Datum
vector_out(PG_FUNCTION_ARGS) {
    Vector *vector = (Vector *) PG_GETARG_POINTER(0);
    char *result;
    char *p;  // 在result上移动
    int i;
    int dim;

    dim = vector->dim;
    /*
	 * Need:
	 *
	 * dim * (FLOAT_SHORTEST_DECIMAL_LEN - 1) bytes for
	 * float_to_shortest_decimal_bufn
	 *
	 * dim - 1 bytes for separator
	 *
	 * 3 bytes for [, ], and \0
	 */
    result = (char *) palloc(FLOAT_SHORTEST_DECIMAL_LEN * dim + 2);
    p = result;
    // 先在开头加上'{'
    *p++ = '{';

    // 计算所需字符串长度
    for (i = 0; i < dim; i++) {
        p += float_to_shortest_decimal_bufn(vector->arr[i], p);
        if (i != dim - 1) {
            *p++ = ',';
        }
    }

    *p++ = '}';
    *p = '\0';

    PG_RETURN_CSTRING(result);
}

/*****************************************************************************
 * New Operators
 *
 * A practical Vector datatype would provide much more than this, of course.
 *****************************************************************************/

PG_FUNCTION_INFO_V1(vector_show_dim);

Datum
vector_show_dim(PG_FUNCTION_ARGS) {
//    Vector *vector = (Vector *) PG_GETARG_POINTER(0);
    Vector *vector = ((Vector *) PG_DETOAST_DATUM(PG_GETARG_DATUM(0)));
    PG_RETURN_INT32(vector->dim);
}

PG_FUNCTION_INFO_V1(vector_calc_l1);

Datum
vector_calc_l1(PG_FUNCTION_ARGS) {
    Vector *v1 = (Vector *) PG_GETARG_POINTER(0);
    Vector *v2 = (Vector *) PG_GETARG_POINTER(1);
    float result = 0.0;
    int32 dim;
    int i;

    // 判断维度是否相同
    if (v1->dim != v2->dim) {
        ereport(ERROR,
                (errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
                        errmsg("Two vectors have different dimensions.")));
    }

    dim = v1->dim;

    for (i = 0; i < dim; i++) {
        result += (v1->arr[i] - v2->arr[i]) * (v1->arr[i] - v2->arr[i]);
    }

    result = sqrtf(result);
    PG_RETURN_FLOAT4(result);
}

PG_FUNCTION_INFO_V1(vector_minus);

Datum
vector_minus(PG_FUNCTION_ARGS) {
    Vector *v1 = (Vector *) PG_GETARG_POINTER(0);
    Vector *v2 = (Vector *) PG_GETARG_POINTER(1);
    Vector *result;
    int32 dim;
    int i;

    // 判断维度是否相同
    if (v1->dim != v2->dim) {
        ereport(ERROR,
                (errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
                        errmsg("Two vectors have different dimensions.")));
    }

    dim = v1->dim;

    result = (Vector *) palloc0(VARHDRSZ + sizeof(int32) + dim * sizeof(float));
//    result->arr = (float *) palloc0(dim * sizeof(float));
    // 记得要SET_VARSIZE
    SET_VARSIZE(result, VARHDRSZ + sizeof(int32) + dim * sizeof(float));
    result->dim = dim;

    for (i = 0; i < dim; i++) {
        result->arr[i] = v1->arr[i] - v2->arr[i];
    }

    PG_RETURN_POINTER(result);
}

PG_FUNCTION_INFO_V1(vector_add);

Datum
vector_add(PG_FUNCTION_ARGS) {
    Vector *v1 = (Vector *) PG_GETARG_POINTER(0);
    Vector *v2 = (Vector *) PG_GETARG_POINTER(1);
    Vector *result;
    int32 dim;
    int i;

    // 判断维度是否相同
    if (v1->dim != v2->dim) {
        ereport(ERROR,
                (errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
                        errmsg("Two vectors have different dimensions.")));
    }

    dim = v1->dim;

    result = (Vector *) palloc0(VARHDRSZ + sizeof(int32) + dim * sizeof(float));
//    result->arr = (float *) palloc0(dim * sizeof(float));
    // 记得要SET_VARSIZE
    SET_VARSIZE(result, VARHDRSZ + sizeof(int32) + dim * sizeof(float));
    result->dim = dim;

    for (i = 0; i < dim; i++) {
        result->arr[i] = v1->arr[i] + v2->arr[i];
    }

    PG_RETURN_POINTER(result);
}
