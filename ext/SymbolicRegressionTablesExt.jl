module SymbolicRegressionTablesExt

using Tables: Tables
import SymbolicRegression.MLJInterfaceModule:
    _tables_istable,
    _tables_colnames,
    _tables_columns,
    _tables_matrix,
    _tables_table,
    is_extension_loaded

is_extension_loaded(::Val{:Tables}) = true

_tables_istable(X) = Tables.istable(X)
_tables_colnames(X) = collect(Symbol, Tables.columnnames(Tables.columns(X)))
_tables_columns(X) = Tables.columns(X)
_tables_matrix(X; transpose::Bool=false) = Tables.matrix(X; transpose)
function _tables_table(out_matrix::AbstractMatrix; names, prototype)
    header = Symbol.(names)
    matrix_table = Tables.table(out_matrix; header)
    prototype === nothing && return matrix_table
    return Tables.materializer(prototype)(matrix_table)
end

end
