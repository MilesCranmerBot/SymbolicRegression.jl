module SymbolicRegressionTablesExt

using Tables: Tables
import SymbolicRegression.MLJInterfaceModule:
    _tables_istable, _tables_colnames, _tables_matrix, _tables_table

_tables_istable(X) = Tables.istable(X)
_tables_colnames(X) = collect(Symbol, Tables.columnnames(Tables.columns(X)))
_tables_matrix(X; transpose::Bool=false) = Tables.matrix(X; transpose)
function _tables_table(out_matrix::AbstractMatrix; names, prototype)
    header = Symbol.(names)
    return Tables.materializer(prototype)(Tables.table(out_matrix; header))
end

end
