module MaxDegreeDiGraph

import Graphs, Graphs.SimpleGraphs

abstract type AbsMaxDegDiGraph{T} <: Graphs.AbstractGraph{T} end

"""
    MaxDegDiGraph(n, d)

A directed graph with `n` vertices and a maximum allowed out-degree of `d` for 
each vertex.

The graph structure is implemented internally as an `n x d` matrix where 
`adj_matrix[j, u]` stores the index of the j-th target vertex connected from 
source vertex `u`. Empty slots are represented by 0.

Vertices are 1-indexed.
"""
struct MaxDegDiGraph{T} <: AbsMaxDegDiGraph{T}
    n::T            # Number of vertices (rows in the adjacency matrix)
    d::T            # Maximum out-degree (columns in the adjacency matrix)
    M::Matrix{T} # n x d matrix storing neighbor indices (0 for no edge)
end

struct MDDGEdgeIter{T}
    M::Matrix{T}
    dims::Tuple{Int,Int}
end

MDDGEdge{T} = Graphs.SimpleDiGraphEdge{T}

Graphs.edgetype(::MaxDegDiGraph{T}) where {T} = Graphs.SimpleDiGraphEdge{T}

MDDGEdgeIter( G::MaxDegDiGraph{T} ) where{T} = MDDGEdgeIter{T}( G.M, size( G.M ) )

Graphs.edges( G::MaxDegDiGraph{T} ) where{T} = MDDGEdgeIter( G )


function  next( iter::MDDGEdgeIter{T}, state::Tuple{Int,Int} ) where {T}
    (row, col) = state
    while true
        row += 1
        if  row > iter.dims[ 1 ]
            row = 1
            col += 1
            if col > iter.dims[ 2 ]
                return  nothing
            end
        end
        if  ( ! iszero( iter.M[ row, col ] ) )
            return  (row, col)
        end
    end    
end

function Base.iterate(iter::MDDGEdgeIter{T} )  where {T}
    state = next( iter, ( 0, 1 ) )
    if  state == nothing
        return  nothing
    end
    
    return  (MDDGEdge{T}( state[ 2 ], iter.M[ state... ] ), state ) 
end

function Base.iterate(iter::MDDGEdgeIter{T}, state::Tuple{Int, Int} ) where {T}
    nstate = next( iter, state )
    if  nstate == nothing
        return  nothing
    end
    
    return  (MDDGEdge{T}( state[ 2 ], iter.M[ nstate... ] ), nstate ) 
end



# --- Constructor ---

function MaxDegDiGraph(n::Int, d::Int)
    if n <= 0 || d < 0
        error("Number of vertices (n) must be positive and "
             * "maximum degree (d) must be non-negative.")
    end
    # Initialize the n x d adjacency matrix with zeros (sentinel for no edge)
    M = zeros(Int, d, n)
    return MaxDegDiGraph{Int}(n, d, M)
end

# Implementation of functions for AbstractBoundedDegreeGraph
function Base.show(io::IO, g::MaxDegDiGraph )
    dir =  "directed";
    return print(io, "{$(nv(g)), $(ne(g))} $dir $(typeof(g)) graph with maximum degree $(g.degree)")
end


@inline function order_edge(g::MaxDegDiGraph, i, j) 
    if is_directed(g) 
        return (i, j)
    else
        return minmax(i, j)
    end
end

@inline order_edge(g::MaxDegDiGraph, e) = order_edge(g, src(e), dst(e))

# --- Graph Interface Methods ---

"""
    nv(g::MaxDegDiGraph)

Returns the number of vertices in the graph.
"""
Graphs.nv(g::MaxDegDiGraph) = g.n


"""
    ne(g::MaxDegDiGraph)

Returns the current number of edges in the graph.
"""
function Graphs.ne(g::MaxDegDiGraph)
    # Count all non-zero entries in the adjacency matrix
    return count(x -> x != 0, g.M)
end

"""
    is_directed(g::MaxDegDiGraph)

Returns true, as this is a directed graph structure.
"""
Graphs.is_directed(::Type{MaxDegDiGraph}) = true
Graphs.is_directed(g::MaxDegDiGraph) = true

"""
    outneighbors(g::MaxDegDiGraph, u::Int)

Returns a list of vertices v such that an edge (u, v) exists.
"""
function Graphs.outneighbors(g::MaxDegDiGraph, u::Int)
    if !(1 <= u <= g.n)
        error("Vertex $u is out of bounds (1 to $(g.n))")
    end
    
    # Filter out the sentinel value (0) from the row corresponding to vertex u
    #return sort( filter(v -> v != 0, g.M[u, :]) )
    return  filter(v -> v != 0, g.M[:,u]) 
end

"""
    add_edge!(g::MaxDegDiGraph, u::Int, v::Int)

Attempts to add a directed edge from vertex u to vertex v.
Returns true if the edge was added, false otherwise.
"""
function Graphs.add_edge!(g::MaxDegDiGraph{T}, u::T, v::T ) where{T}
    if !( ( 1 <= u <= g.n)  &&  ( 1 <= v <= g.n ) )
        error("Vertices (u, v) are out of bounds (1 to (g.n))")
    end
    if u == v
        @warn "Self-loops are currently supported but generally discouraged in simple graphs."
    end
    col = view( g.M, :, u )
    for j in 1:g.d
        ent = col[ j ]
        if  ent == v
            #@warn "Edge ($u, $v) already exists."
            return false
        end
        if ent == 0
            # Slot found: add the neighbor and return true
            col[ j ] = v
            return true
        end
    end
    
    # 3. No empty slot found (max out-degree d exceeded)
    @warn "Failed to add edge ($u, $v). Vertex $u already has the maximum out-degree ($(g.d))."
    @error "Failed to add edge ($u, $v). Vertex $u already has the maximum out-degree ($(g.d))."
    println( "Max graph degree: ", g.d )

    @assert( false )
    return false
end

function Graphs.has_edge(g::MaxDegDiGraph, i::Int, j::Int ) 
    #i, j = order_edge(g, e...)

    col = view( g.M, :, i )
    #col = g.M[:, i ]
    for t ∈ 1:g.d
        if   col[t] == j
            return true
        end
    end
    
    return  false
end

Graphs.vertices(g::MaxDegDiGraph) = 1:nv(g)

Graphs.has_vertex(g::MaxDegDiGraph, i) = i in vertices(g)

function Graphs.add_vertices!(g::MaxDegDiGraph{T}, n::T) where {T<:Integer} 
    n_old = nv(g)
    
    if n <= 0
        return nothing 
    end

    M = zeros(T, g.d, g.n + n)

    A = g.M
    # Copy the data from A into the top-left corner of B
    M[1:size(A, 1), 1:size(A, 2)] = A
    
    g.M = M;
end


"""
    rem_edge!(g::MaxDegDiGraph, u::Int, v::Int)

Removes the directed edge from vertex u to vertex v.
Returns true if the edge was removed, false otherwise.
"""
# e::SimpleDiGraphEdge{T}
function Graphs.rem_edge!(g::MaxDegDiGraph{T}, u::Int,v::Int) where{T}
#    u,v = T.(Tuple(e))a
    if !(1 <= u <= g.n && 1 <= v <= g.n)
        error("Vertices ($u, $v) are out of bounds (1 to $(g.n))")
    end

    # Find the target vertex v in the row for vertex u
    col = view( g.M, :, u )
    #col = g.M[ :, u]
    
    for j in g.d:-1:1
        if col[j] == v
            # Edge found: replace the target with the sentinel value (0)
            col[ j ] = 0
            return true
        end
    end

    @warn "Edge ($u, $v) does not exist."
    return false
end

#function rem_edge!(g::MaxDegDiGraph{T}, u::Integer, v::Integer) where {T}
#    return rem_edge!(g, edgetype(g)(T(u), T(v)))
#end


#add_edge!(g::MaxDegDiGraph, x) = add_edge!(g, edgetype(g)(x))
#add_edge!(g::MaxDegDiGraph, x, y) = add_edge!(g, edgetype(g)(x, y))



Graphs.zero(::Type{T}) where {T <: MaxDegDiGraph} = T(0, 1)


function   Graphs.inneighbors(g::MaxDegDiGraph, dst)
    out = Vector{T}()
    for i ∈ 1:nv(g)
        for  b ∈ 1:g.d
            ent = g.M[ b, i ]
            if  ent == dst
                push!( out, i )
                break
            end
        end
    end
    #sort!( out )

    return  out
end


export MaxDegDiGraph #, nv, ne, is_directed, outneighbors, add_edge!, rem_edge!

end # module MaxDegDiGraph

