CREATE EXTENSION pg_dirtyread;
show txn_id;

CREATE TABLE foo (a int, b int);

ALTER TABLE foo SET (
    autovacuum_enabled = false, toast.autovacuum_enabled = false
);

INSERT INTO foo VALUES (0, 0);

SELECT xmin, * from foo;

INSERT INTO foo VALUES (1, 1);

SELECT xmin, * from foo;

INSERT INTO foo VALUES (2, 2);

SELECT xmin, * from foo;

UPDATE foo
SET b = b + 2
WHERE a != 0;

SELECT xmin, * from foo;

CREATE OR REPLACE FUNCTION test() RETURNS setof foo AS $$
DECLARE
t integer;
rec record;
BEGIN
    select xmin from foo where a = 0 into t;
    FOR  i in 0..4 LOOP
        EXECUTE format('SET txn_id = %s', t+i);
        RAISE NOTICE 'txn_id is %', t+i;
        FOR rec in  select * from dirtyread('foo') as t(a int, b int) LOOP
            return next rec;
        END LOOP;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

select test();

DROP FUNCTION test;
DROP TABLE foo;
DROP EXTENSION pg_dirtyread;
