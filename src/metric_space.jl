#! /bin/env julia

push!(LOAD_PATH, pwd()*"/src/cg/")
push!(LOAD_PATH, pwd()*"/src/" )

using FrechetDist
using FrechetDist.cg
using FrechetDist.cg.polygon
using FrechetDist.cg.point
using Graphs
using Distances
using DataStructures

abstract type AbstractFMetricSpace end
"""
    metric(space::AbstractFMetricSpace, x, y)

Computes the distance between points `x` and `y` in the given `space`.
"""
function metric end

"""
    size( space::AbstractFMetricSpace )

   Return the number of points in the finite metric space (i.e., n).
"""
#function size end


struct PointsSpace{PType} <: AbstractFMetricSpace
    #n::Int  # Number of points
    P::Vector{PType}
end

function metric( P::PointsSpace{PType}, x, y ) where {PType}
    if  ( x == y )
        return  0.0;
    end
    return  Dist( P.P[ x ], P.P[ y ] );
end

function Base.size(P::PointsSpace{PType} ) where {PType}
    return  length( P.P );
end


####################################################################
# MPointsSpace: Metric space defined by a matrix.
# Columns are the points.
####################################################################
struct MPointsSpace{PType} <: AbstractFMetricSpace
    #n::Int  # Number of points
    m::Matrix{PType}
end


function metric( P::MPointsSpace{PType}, x, y ) where {PType}
    if  ( x == y )
        return  0.0;
    end
    return  euclidean( P.m[ :, x ], P.m[ :, y ] );
end

function Base.size(P::MPointsSpace{PType} ) where {PType}
    return  size( P.m, 2 );
end

####################################################################
# PermutMetric: A permutation of a finite metric space.
####################################################################
struct PermutMetric{MetricType} <: AbstractFMetricSpace
    n::Int  # Number of points
    m::MetricType;
    I::Vector{Int64};
end

function  PermutMetric(_m::MetricType ) where {MetricType}
    _n = size( _m );
    return PermutMetric( _n, _m, [i for i ∈ 1:_n] );
end

function  PermutMetric(_m::MetricType, _I::Vector{Int64} ) where {MetricType}
    return PermutMetric{MetricType}( size( _m ), _m, _I );
end


function metric( P::PermutMetric{MetricType}, x, y ) where {MetricType}
    if  ( x == y )
        return  0.0;
    end
    return  metric( P.m, P.I[ x ], P.I[ y ] );
end

function Base.size(P::PermutMetric{MetricSpace} ) where {MetricSpace}
    return  P.n;
end

function swap!( P::PermutMetric{MetricType}, x, y ) where {MetricType}
    if  x == y then
        return;
    end
    I[ x ], I[ y ] = I[ y ], I[ x ]
end

######################################################################

function  update_distances( M::AbstractFMetricSpace, I, D, pos, n )
    x = I[ pos ];
    for  i ∈ pos+1:n
        y = I[ i ];
        d = metric( M, x, y )
        #println( "d: ", d, "  x: ", x, "  y: ", y, "  pos: ", pos, "  i: ", i );
        if   ( ( d < D[ i ] )  ||  ( D[ i ] < zero(Float64) ) )
            D[ i ] = d;
        end
    end
end


function  greedy_permutation_naive( M::AbstractFMetricSpace )
    n = size( M );
    println( "m: ", n );
    I = [i for i ∈ 1:n ];
    D = fill( Float64(-1.0), n );

    # Initialize
    update_distances( M, I, D, 1, n );

    for  i ∈ 2:n
        #println( "I : ", i );
        pos = argmax( D[ i:n ] ) + i - 1;
        D[ i - 1 ] = D[ pos ];
        I[ i ], I[ pos ] =  I[ pos ], I[ i ];
        #Centers[ i ], Centers[ pos ] =  Centers[ pos ], Centers[ i ];
        D[ i ], D[ pos ] =  D[ pos ], D[ i ];
        update_distances( M, I, D, i, n );
    end
    D[ n ] = zero( Float64 );
    
    return  I, D
end



"""
    read_fvecs(filename::String)

[Written by gemini.]
Reads a binary .fvec file and returns a matrix where each column is a vector.
The .fvec format consists of a 4-byte integer for the number of dimensions,
followed by the single-precision floating-point vector data. This pattern
repeats for each vector in the file.
"""
function read_fvecs(filename::String)
    try
        # Open the file in binary mode for reading
        io = open(filename, "r")
        # Ensure the file is closed automatically even if an error occurs
        finalizer(close, io)

        # Read the number of dimensions from the first 4 bytes (as an Int32)
        if eof(io)
            throw(ErrorException("File is empty or malformed."))
        end
        d = read(io, Int32)

        # Basic format check: dimension should be a positive integer
        if d <= 0
            throw(ErrorException("Invalid vector dimension read from file: d = $d."))
        end

        # Calculate the size of a single vector record in bytes
        vec_record_size_bytes = sizeof(Int32) + d * sizeof(Float32)

        # Seek to the end of the file to determine its total size
        seekend(io)
        file_size_bytes = position(io)

        # Check if the file size is a multiple of the expected record size
        if file_size_bytes % vec_record_size_bytes != 0
            throw(ErrorException("File size is not a multiple of the vector record size, indicating a malformed file."))
        end

        # Rewind to the beginning of the file to start reading vectors
        seekstart(io)

        # Calculate the total number of vectors in the file
        num_vectors = div(file_size_bytes, vec_record_size_bytes)

        # Pre-allocate a matrix to store the vectors for efficiency.
        # The dimensions are (d, num_vectors), where each column is a vector.
        data = zeros(Float32, d, num_vectors)

        for i in 1:num_vectors
            # Read the number of dimensions for the current vector
            # This value should be consistent with the first one we read.
            current_d = read(io, Int32)
            if current_d != d
                throw(ErrorException("Inconsistent vector dimension detected at vector $i: expected $d, but got $current_d."))
            end

            # Read the d single-precision floats and store them in the current column of the matrix
            read!(io, view(data, :, i))
        end

        return data
    catch e
        # Catch and re-throw with a more descriptive error message
        rethrow(ErrorException("Error reading .fvec file: $(e.msg)"))
    end
end


###################################################################

#mutable struct NNGraph{FMS} where {FMS <: AbstractFMetricSpace}
mutable struct NNGraph{FMS <: AbstractFMetricSpace}
    n::Int64
    m::FMS
    G::DiGraph
end

function  NNGraph( _m::FMS ) where{FMS}
    _n = size( _m );
    return NNGraph{FMS}( _n, _m, DiGraph( _n ) );
end        

"""
    nng_random_dag(_G::NNGraph{FMS}, d::Int64 = 10) where {FMS}

Generates a random directed acyclic graph (DAG) for the given nearest neighbor graph (`NNGraph`).

# Arguments
- `_G::NNGraph{FMS}`: The nearest neighbor graph object, where `FMS` is a finite metric space.
- `d::Int64`: The maximum number of outgoing edges (degree) for each vertex. Defaults to `10`.

# Behavior
- For each vertex `i` in the graph, the function randomly selects up to `d` vertices from the set of vertices with indices greater than `i` (to ensure the graph remains acyclic).
- The selected vertices are added as outgoing edges from vertex `i`.
- Duplicate edges are removed using `unique!`.
"""
function nng_random_dag( _G::NNGraph{FMS}, d::Int64 = 10 ) where {FMS}
    G = _G.G;
    n = _G.n;
    for  i ∈ 1:(n-1)
        dst = rand( (i+1):n, d );
        unique!( dst );
        for j ∈ dst
            add_edge!( G, i, j )
        end
    end
end

function find_k_lowest_f(G::DiGraph, u::Int, k::Int, f )
    
    # Use a PriorityQueue to store vertices to visit, prioritized by their f value.
    # The lowest f value has the highest priority.
    pq     = PriorityQueue{Int, Float64}()
    result = PriorityQueue{Int, Float64}(Base.Reverse)
    enqueue!(pq, u, f( u ) )

    # A set to keep track of visited vertices to avoid cycles and redundant processing.
    visited = Set{Int}()
    push!(visited, u)
    #println( "f(u) = ", f(u ) )
    #println( typeof( f(u) ) );
    
    enqueue!( result, u, f( u ) );
    #visited = Set{Int}()
    #push!( queued, u)
    cost::Int = 0;
    # Loop until the PriorityQueue is empty or we have found k vertices.
    while !isempty(pq)
        # Dequeue the vertex with the current lowest f value.
        #println( "before dequeue!" );
        v, v_val = peek( pq );
        dequeue!(pq)
        cost += 1;
        #println( "cost: ", cost );

        ℓ = length( pq )
        if  ( ℓ >= k )
            while  ( length( result ) > k )
                dequeue!( result );
            end
            r, r_val = peek( result )

            # Maybe too aggressive?
            if  ( v_val > r_val )
                break;
            end
        end

        # Put it into the results...
        result[ v ] = v_val;

        N = outneighbors(G, v)

        for  o ∈ N
            cost += 1;
            if  ( o ∈ visited )
                continue
            end
            o_val = f( o );
            enqueue!( pq, o, o_val );
            push!( visited, o );
        end
    end

    #println( "Done?" );
    cl = collect( result );
    #println( "MSHOGI: ", typeof( cl ) );
    rs = [cl[i]  for i ∈ length(cl):-1:1]
    #   reverse( cl );
    #=
    println( "cl: " );
    println( cl );
    println( "rs: " );
    println( rs );
    println( " SHOGI" );
    exit( -1 );
    =#
    return   rs, cost
end

function nng_greedy_dag( _G::NNGraph{FMS}, 
                         R::Vector{Float64},
                         factor::Float64 = 1.1 
                        ) where {FMS}
    G = _G.G;
    n = _G.n;
    @assert( length( R ) == n, "R must have the same length ." );
    for i ∈ 2:n
        result, _ = find_k_lowest_f( G, i, 20, j -> metric( _G.m, i, j) );
        for  (u, dist) ∈ result
            if  ( dist < factor * R[ i ] )
                add_edge!( G, u, i );
            end
        end
    end

end                        


function  (@main)(args)

    if  ( length( args ) > 0 )
        m = read_fvecs( args[ 1 ] );
        d::Int64 = size( m, 1 );
        n::Int64 = size( m, 2 );
        println( "Dimension: ", d );
        println( "n        : ", n );
        
        PS = MPointsSpace( m );
        I, D = greedy_permutation_naive( PS )

        println( "I  (type): ", typeof( I ) )
        println( "PS (type): ", typeof( PS ) )
        m = PermutMetric( PS, I );
        GA = NNGraph( m );
        GB = NNGraph( m );
        nng_random_dag( GA, 10 );

        nng_greedy_dag( GB, D, 2.1 )
        println( "# edges random(10) : ", ne(GA.G) );
        println( "# edges greedy     : ", ne(GB.G) );
        #=
        for i ∈ 1:n
            println( i, ":", I[i], "  D: ", D[i ] );
        end
        =#
        
        exit
    end

    
    n = 10;
    P = Polygon_random( 2, Float64, n );

    println( length( args ) )

    println( "Hello world!" );
    PS = PointsSpace( Points( P ) );
    I, D = greedy_permutation_naive( PS )

    println( typeof( I ) )
    println( typeof( D ) )
    #=
    for i ∈ 1:n
        println( i, ":", I[i], "  D: ", D[i ] );
    end
    =#
    
    return  0;
end

