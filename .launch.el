;;; Code;
(dap-register-debug-template
  "codelldb::Debug"
  (list :type "codelldb"
        :request "launch"
        :name "codelldb::Debug::psql"
        :MIMode "lldb"
        ;;: stopAtEntry 't
        :stopOnEntry 't
        :args ["uni"]
        :program "/Users/jqin/PGDev/postgresql-12.5/src/bin/psql/psql"
        :cwd "/Users/jqin/PGDev/postgresql-12.5/"))

(dap-register-debug-template
  "codelldb::DebugAttach"
  (list :type "codelldbattach"
        :request "attach"
        :name "codelldb::DebugAttach::postgres"
        :pid 54488
        :MIMode "lldb"
        :program "/Users/jqin/PGDev/postgresql-12.5/src/backend/postgres"
        :cwd "/Users/jqin/PGDev/postgresql-12.5/"))
