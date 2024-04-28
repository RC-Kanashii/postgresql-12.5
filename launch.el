;; Eval Buffer with `M-x eval-buffer' to register the newly created template.

(dap-register-debug-template
  "LLDB::Run"
  (list :type "lldb"
        :request "launch"
        :name "LLDB::Run"
        :target "/Users/jqin/PGDev/postgresql-12.5/src/backend/postgres"
        :cwd "/Users/jqin/PGDev/postgresql-12.5/"))
