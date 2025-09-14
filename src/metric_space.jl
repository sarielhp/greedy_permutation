#! /bin/env julia

#using DataStructures: cleanup!

push!(LOAD_PATH, pwd()*"/src/cg/")
push!(LOAD_PATH, pwd()*"/src/" )

using FrechetDist
using FrechetDist.cg
using FrechetDist.cg.polygon
using FrechetDist.cg.point
using Graphs
using Distances
using DataStructures
using Printf
using Random

using PlotGraphviz

using SimpleWeightedGraphs

using TimerOutputs

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

function dist( P::PointsSpace{PType}, x, y ) where {PType}
    if  ( x == y )
        return  0.0;
    end
    return  Dist( P.P[ x ], P.P[ y ] );
end

function dist_real( P::PointsSpace{PType}, x, y_real ) where {PType}
    return  Dist( P.P[ x ], y_real );
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


function dist( P::MPointsSpace{PType}, x, y ) where {PType}
    if  ( x == y )
        return  0.0;
    end
    return  euclidean( P.m[ :, x ], P.m[ :, y ] );
end

function dist_real( P::MPointsSpace{PType}, x, y_real ) where {PType}
    return  euclidean( P.m[ :, x ], y_real );
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


function dist( P::PermutMetric{MetricType}, x::Int, y::Int ) where {MetricType}
    if  ( x == y )
        return  0.0;
    end
    return  dist( P.m, P.I[ x ], P.I[ y ] );
end

function dist_real( P::PermutMetric{MetricType}, x, y_real ) where {MetricType}
    return  dist_real( P.m, P.I[ x ], y_real );
end

function Base.size(P::PermutMetric{MetricSpace} ) where {MetricSpace}
    return  P.n;
end

function swap!( P::PermutMetric{MetricType}, x, y ) where {MetricType}
    if  x == y
        return;
    end
    P.I[ x ], P.I[ y ] = P.I[ y ], P.I[ x ]
end

######################################################################

function  update_distances( M::AbstractFMetricSpace, I, D, pos, n )
    x = I[ pos ];
    for  i ∈ pos+1:n
        y = I[ i ];
        d = dist( M, x, y )
        #println( "d: ", d, "  x: ", x, "  y: ", y, "  pos: ", pos, "  i: ", i );
        if   ( ( d < D[ i ] )  ||  ( D[ i ] < zero(Float64) ) )
            D[ i ] = d;
        end
    end
end


function  greedy_permutation_naive( M::AbstractFMetricSpace, n::Int64 )
    #n = size( M );
    println( "m: ", n );
    I = [i for i ∈ 1:size(M) ];   ## Imprtant: n might be smaller than size(M)
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
    desc::String
end

function  NNGraph( _m::FMS, _n::Int64 ) where{FMS}
#    _n = size( _m );
    return NNGraph{FMS}( _n, _m, DiGraph( _n ), "" );
end        

function  description!( G::NNGraphF, _desc::String ) where {NNGraphF}
    G.desc = _desc;
end

"""
    nng_random_dag!(_G::NNGraph{FMS}, d::Int64 = 10) where {FMS}

Generates a random directed acyclic graph (DAG) for the given nearest neighbor graph (`NNGraph`).

# Arguments
- `_G::NNGraph{FMS}`: The nearest neighbor graph object, where `FMS` is a finite metric space.
- `d::Int64`: The maximum number of outgoing edges (degree) for each vertex. Defaults to `10`.

# Behavior
- For each vertex `i` in the graph, the function randomly selects up to `d` vertices from the set of vertices with indices greater than `i` (to ensure the graph remains acyclic).
- The selected vertices are added as outgoing edges from vertex `i`.
- Duplicate edges are removed using `unique!`.
"""
function nng_random_dag!( _G::NNGraph{FMS}, n::Int64, d::Int64 = 10 ) where {FMS}
    G = _G.G;
    for  i ∈ 1:(n-1)
        dst = rand( (i+1):n, d );
        unique!( dst );
        for j ∈ dst
            add_edge!( G, i, j )
        end
    end
end

function nng_random_dag( m::FMS, degree::Int64 ) where {FMS}
    G_rand = NNGraph( m, size( m ) );
    nng_random_dag!( G_rand, size( m ), degree )
    return  G_rand
end


function find_k_lowest_f(G::DiGraph, u::Int, k::Int, f )
    @assert( k > 0 );
    
    cost::Int = 0;

    @inline function  eval_f( z )
        cost += 1
        return f( z );
    end
    
    #println( "find_k_lowest_f" );
    # Use a PriorityQueue to store vertices to visit, prioritized by their f value.
    # The lowest f value has the highest priority.
    pq     = PriorityQueue{Int, Float64}()
    result = PriorityQueue{Int, Float64}(Base.Reverse)
    u_val = eval_f( u );

    enqueue!( pq, u, u_val )
    enqueue!( result, u, u_val );
    
    #println( "u_val: ", u_val );
    
    # A set to keep track of visited vertices to avoid cycles and redundant processing.
    visited = Set{Int}()
    push!(visited, u)
    
    f_threshold = false;
    threshold = -10.0;

    # Loop until the PriorityQueue is empty or we have found k vertices.
    while !isempty(pq)
        # Dequeue the vertex with the current lowest f value.
        #println( "before dequeue!" );
        v, v_val = peek( pq );
        dequeue!(pq)

        ℓ = length( result )
        if  ( ℓ >= k )
            f_threshold = true;
            while  ( length( result ) > k )
                dequeue!( result );
            end
            _, threshold = peek( result )
            #println( "threshold: ", threshold );
            # Maybe too aggressive?
            if  ( v_val > threshold )
                break;
            end
        end

        # Put it into the results...
        result[ v ] = v_val;

        N = outneighbors(G, v)

        for  o ∈ N
            if  ( o ∈ visited )
                continue
            end

            push!( visited, o );

            o_val = eval_f( o );
            # @assert( o_val > 0.0 );
            if  ( f_threshold   &&  ( threshold < o_val ) )
                continue
            end
            
            enqueue!( pq, o, o_val );
            enqueue!( result, o, o_val );
        end
    end

    cl = collect( result );
    rs = [cl[i]  for i ∈ length(cl):-1:1]

    return   rs, cost, visited
end


function find_k_lowest_f_greedy(G::DiGraph, u::Int, k::Int, f )
    @assert( k > 0 );
    
    # Use a PriorityQueue to store vertices to visit, prioritized by their f value.
    # The lowest f value has the highest priority.
    pq     = PriorityQueue{Int, Float64}()
    result = PriorityQueue{Int, Float64}(Base.Reverse)
    leftover = Vector{Int}()
    visited = Set{Int}()
    cost::Int = 0  # The number of times f was called...
    f_threshold = false;
    threshold = -10.0;

    @inline function  eval_f( y )
        cost += 1
        return  f( y )
    end
        
    function  leftover_to_queue()
        while  !isempty( leftover )
            #z, z_val = peek( leftover );
            z = pop!( leftover );
            if  ( z ∈ visited )
                continue
            end
            push!( visited, z );
            @assert( z ∈ visited )
            z_val = eval_f( z )
            if  ( f_threshold   &&  ( threshold < z_val ) )
                continue
            end

            enqueue!( pq, z, z_val );
            enqueue!( result, z, z_val );
        end
    end

    u_val = eval_f( u )

    # Threshold for greedy jump. This constant maybe should be optimized?
    δ = 0.7

    enqueue!( pq, u, u_val )
    enqueue!( result, u, u_val )
    curr = u_val;
    
    # A set to keep track of visited vertices to avoid cycles and
    # redundant processing.
    push!(visited, u)
    
    # Loop until the PriorityQueue is empty or we have found k vertices.
    while ( ( !isempty(pq) )  ||  ( !isempty( leftover ) ) )
        if isempty( pq )
            leftover_to_queue()
            if  isempty( pq )
                break
            end
        end
        
        # Dequeue the vertex with the current lowest f value.
        #println( "before dequeue!" );
        v, v_val = peek( pq );
        dequeue!(pq)

        if  v_val < curr
            curr = v_val;
        end
        
        ℓ = length( result )
        if  ( ℓ >= k )
            f_threshold = true;
            while  ( length( result ) > k )
                dequeue!( result );
            end
            _, threshold = peek( result )
            #println( "threshold: ", threshold );
            # Maybe too aggressive?
            if  ( v_val > threshold )
                break;
            end
        end

        # Put it into the results...
        result[ v ] = v_val;

        N = outneighbors(G, v)

        f_copy = false
        for  o ∈ N
            if  ( o ∈ visited )
                continue
            end
            if  f_copy
                push!( leftover, o )
                continue
            end

            push!( visited, o );

            o_val = eval_f( o );
            # @assert( o_val > 0.0 );
            if  ( f_threshold   &&  ( threshold < o_val ) )
                continue
            end

            enqueue!( pq, o, o_val );
            enqueue!( result, o, o_val );

            # Greedily jump if the improvement is significant...
            if  ( o_val < δ * curr )
                f_copy = true;
                curr = o_val
            end
        end
    end

    cl = collect( result );
    rs = [cl[i]  for i ∈ length(cl):-1:1]

    return   rs, cost, visited
end


function  nn_add_edges!( G, m, i, r_i, factor, max_degree )
    function  dist_to_i( j )
        return  dist( m, i, j )
    end

    out = Vector{Int64}()
    #println( "TYPEOF: ", typeof( m ) );
    result, _ = find_k_lowest_f(G, 1, 200, dist_to_i )
    forced::Int64 = 3 #max_degree/2 #3;
    for (u, dista) ∈ result        
        if (dista < factor * r_i ) ||  ( forced > 0 )
            forced -= 1;
            add_edge!(G, u, i)
            push!( out, u )
        end
    end

    return  out
end

function nng_greedy_dag!(_G::NNGraph{FMS},
    R::Vector{Float64},
    n::Int64,
    factor::Float64=1.1
) where {FMS}
    G = _G.G
    m = _G.m;
    k = 200;
    @assert(length(R) == n, "R must have the same length .")
    for i ∈ 2:n
        result, _ = find_k_lowest_f(G, 1, k, j -> dist( m, i, j))
        edges_added = 0;
        forced = 2;
        count = 0;
        for (u, dist) ∈ result
            count += 1;
            #f_add = ( count == 1 )  ||  
            # the i-1 is important in the following line! It is the true distance.
            if ( (dist < factor * R[ i - 1 ])  ||  ( forced > 0 ) )
                edges_added += 1;
                forced -= 1;
                add_edge!(G, u, i)
            end
        end

        if  ( edges_added == 0 )
            dst = dist( m, 1, i );
            for j = 1:i-1
                dst = min( dst, dist( m, i, j) );
            end
            println( "No edges added for : ", i );
            println( result );
            println( "Real dist: ", dst );
            println( R[ i - 1 ] );
            println( R[ i ] );
            exit( -1 );
        end
    end
end


function  nng_greedy_graph( m_i, I, D, factor );
    n = size( m_i, 2 );
    PS = MPointsSpace( m_i );
    m = PermutMetric( PS, I );
    G = NNGraph( m, n );

    nng_greedy_dag!( G, D, n, factor )
    return  G
end


function  nng_nn_search_all( _G::NNGraph{FMS}, 
                                 q::Int,
                                 k::Int = 20
                                ) where {FMS}
    result, _, visited = find_k_lowest_f( _G.G, 1, k, j -> dist( _G.m, j, q ) );

    return  result, visited
end

function  nng_nn_search_reg( _G::NNGraph{FMS}, 
                                 q::Int,
                                 k::Int = 20
                                ) where {FMS}
    result, _ = nng_nn_search_all( _G, q, k );

    return  result[ 1 ];
end

function  nng_nn_search( _G::NNGraph{FMS}, 
                                 q_real,
                                 k::Int = 20
                                ) where {FMS}
    result, cost = find_k_lowest_f( _G.G, 1, k, j -> dist_real( _G.m, j, q_real) );
    
    return  result[ 1 ];
end

function  nng_nn_search_print( _G::NNGraphFMS, 
                               q_real,
                               k::Int,
                               ℓ::Float64
                                ) where {NNGraphFMS}

    result_g, cost_g = find_k_lowest_f_greedy( _G.G, 1, k,
                                         j -> dist_real( _G.m, j, q_real) );
    _,dist_g = result_g[1];
    f_exact_g = ( dist_g == ℓ )
    s_dist_g = f_exact_g ? "Exact" : @sprintf( "%12g", dist_g );
    
    str = @sprintf( "k: %4d: cost: %7d   d: %s", k, cost_g,
                    s_dist_g  )
    return str, f_exact_g, [Float64(k), Float64( cost_g ) ];
end


"""
Build the nearest-neighbor search graph together with its permutation.
The permutation is encoded in the permutation metric (i.e., m(i) is
the i point in the greedy permutation).
"""
function nng_build_and_permut!(
    m::PermutMetric{FMS},
    factor::Float64 = 1.1,
    f_prune = false,
    α::Float64 = 0.0,
    max_degree = 0
) where {FMS}
    n = size( m )
    sqrt_n = round( Int64, sqrt( n ) )
    G = NNGraph( m, n );

    R = fill( -1.0, n );
    R[ 1 ] = 0.0;

    # queue + radii 
    qadii = PriorityQueue{Int, Float64}(Base.Reverse)
    for i ∈ 2:n
        R[ i ] = dist( m, 1, i )  # i == π[ i ] so redundant to use π 
        enqueue!( qadii, i, R[ i ] )
    end

    println( "n = ", n );
    i::Int = 2
    c_count = 0
    i_iter::Int64 = 0
    handled::Int64 = 0;
    while  ( !isempty( qadii ) )
        i_iter += 1
        curr, r_curr = peek( qadii )
        _, new_rad = nng_nn_search_reg( G, curr );
        #println( "Okay!" );
        if  ( new_rad < 0.99 * r_curr )
            qadii[ curr ] = r_curr = new_rad

            # Maybe it did not change after all
            ncurr, _ = peek( qadii )
            if  ( ncurr != curr )
                c_count += 1;
                #                println( "CONTINUE!" );
                continue;
            end
        end

        # Okay, the point curr is the next point in the greedy permutation...
        swap!( m, i, curr )

        # A bit of shenanigan with the queue
        r_i = qadii[ i ]
        qadii[ curr ] = r_i;
        dequeue!( qadii, i );

        handled += 1;
        # Now we need to add the edges to the DAG...
        N = nn_add_edges!( G.G, m, i, r_i, factor, max_degree )

        # Pruning is done only after we constructed at least half the graph...
        if  ( f_prune  &&  ( handled > (n/2) ) )
            for v ∈ N
                if  outdegree( G.G, v ) > 8*max_degree
                    U = Set{Int}( outneighbors( G.G, v ) )
                    #limit = 4 * max_degree;
                    #limit = max_degree;
                    robust_prune_vertex!( G, v, U, α, max_degree );
                end
            end                    
        end
        R[ i ] = r_curr;

        i += 1;
    end

    println( "c_count: ", c_count, " / ", i_iter );
    return  G, R;
end


function  input_sift_small()
    basedir = "data/sift/"
    m_i = read_fvecs( basedir * "small_learn.fvecs" );
    m_q = read_fvecs( basedir * "small_query.fvecs" );
    return  m_i, m_q;
end

function  input_sift()
    basedir = "data/sift/"
    m_i = read_fvecs( basedir * "learn.fvecs" );
    m_q = read_fvecs( basedir * "query.fvecs" );
    return  m_i, m_q;
end


function  input_random( n, n_q, d )
    return   rand( d, n ), rand( d, n_q );
end


function  nn_search_brute_force( m_i, q_real )::Float64
    dst::Float64 = euclidean( m_i[ :, 1 ], q_real );
    n = size( m_i, 2 );
    for  j ∈ 2:n
        new_dst = euclidean( m_i[ :, j ], q_real );
        if  ( new_dst < dst )
            dst = new_dst;
            if  ( dst == 0.0 )
                println( "DST: ", euclidean( m_i[ :, j ], q_real )  );
                println( "j: ", j );
            end
        end
    end

    return  dst;
end

"""
    Check that the greedy graph has strong connectivity and all the nearest-neighbors are reachable from the first vertex.
"""
function  check_greedy_dag_failure_slow( m_i, m_q, I, D, factor )
    d::Int64 = size( m_i, 1 );
    n::Int64 = size( m_i, 2 );
    n_q::Int64 = size( m_q, 2 );
    
    PS = MPointsSpace( m_i );
    m = PermutMetric( PS, I );
    GB     = NNGraph( m, n )

    nng_greedy_dag!( GB, D, n, factor )    
    for i ∈ 1:n_q
        real_dist = nn_search_brute_force( m_i, m_q[ :, i] );
            
        dist = nng_nn_search( GB, m_q[ :, i ], n )[ 2 ];

        if  ( dist != real_dist )
            println( "greedy DAG failed for ", factor )
            println( "dist      : ", dist );
            println( "real_dist : ", real_dist );
            println( "Query: ", m_q[ :, i ] );

            G = SimpleWeightedDiGraph( GB.G );
            dot = plot_graphviz( G );
            write_dot_file( G, "test.dot" );

            for  e ∈ edges( GB.G )
                println( e )
            end
            println( D );
            exit( -1 );
        end
    end
    println( "Greedy DAG passed for: ", factor );
end

"""
Given a set of vertices V in the metric of G, the following function
robustly prune it as described in the Disk-Ann paper, iterative adding the closest point to the out-set, while removing its Apollonius ball from V. 
"""
function  robust_prune( G::NNGraphFMS, p, V::Set{Int},
                        α::Float64, max_size ) where {NNGraphFMS}
    pq     = PriorityQueue{Int, Float64}()

    N = deepcopy( V ); #union( Set{Int}( ), V );
    for x ∈ outneighbors(G.G, p)
        push!( N, x )
    end
        
    delete!( N, p );
    
    for  o ∈ N
        enqueue!( pq, o, dist( G.m, p, o ) )
    end

    out = Set{Int}();
    while  ( ! isempty( pq ) )
        v, v_val = peek( pq );
        dequeue!(pq)
        push!( out, v );
        if  ( length( out ) >= max_size )
            break
        end
        del_list = Vector{Int}();
        for xp ∈ pq
            x = xp[1]
            if   α * dist( G.m, v, x ) <= dist( G.m, p, x )
                push!( del_list, x );
            end
        end
        for x ∈ del_list
            delete!( pq, x );
        end
    end
    
    return out
end


function  robust_prune_vertex!( G::NNGraphFMS, p, V::Set{Int},
                           α, max_size ) where {NNGraphFMS}
    out =  robust_prune( G, p, V, α, max_size );

    
    neighbors_to_delete = Set{Int}( collect( outneighbors(G.G, p) ) )
    del_list = setdiff( neighbors_to_delete, out );
    add_list = setdiff( out, neighbors_to_delete );
    
    for t in del_list
        rem_edge!( G.G, p, t)
    end
    for t in add_list
        add_edge!( G.G, p, t)
    end
end


function random_perm_no_selfies( n )
    while  true
        p = randperm(n)

        f_good = true;
        for i ∈ 1:n
            if  p[ i ] == i
                f_good = false;
                break;
            end
        end
        if  f_good
            return  p
        end
    end
end

function generate_directed_random_graph(n::Int, d::Int)
    G = DiGraph(n)

    for  i ∈ 1:d
        p = random_perm_no_selfies( n );
        for  i ∈ 1:n
            if  ! has_edge( G, i, p[ i ] )
                add_edge!( G, i, p[ i ] )
            end
        end
    end

    return G
end

function  da_clean_inner( G, α::Float64, L::Int64, max_degree::Int64,
                          f_prune_curr::Bool, π::Vector{Int64} )
    n = size( G.m )
    for i ∈ 1:n
        if  ( f_prune_curr  &&  ( outdegree( G.G, π[ i ] ) > max_degree ) )
            U = Set{Int}( outneighbors( G.G, π[ i ] ) )
            robust_prune_vertex!( G, π[ i ], U, α, max_degree );
        end
        result, V = nng_nn_search_all( G, π[ i ], L );
        out = robust_prune( G, π[ i ], V, α, max_degree );
        for j ∈ out
            # π[ i ]???? Verify.
            if  ( outdegree( G.G, j ) >= max_degree )
                U = Set{Int}( outneighbors( G.G, j ) )
                push!( U, π[ i ] );
                robust_prune_vertex!( G, j, U, α, max_degree );
            else
                add_edge!( G.G, j, π[ i ] );
            end
        end
    end
    
    return  G
end



function  da_clean( G, α::Float64, L::Int64, max_degree::Int64, f_prune_curr::Bool = false )
    n = size( G.m )
    π = randperm(n)
    return  da_clean_inner( G, α, L, max_degree, f_prune_curr, π );
end

function  da_clean_rev( G, α::Float64, L::Int64, max_degree::Int64, f_prune_curr::Bool = false )
    n = size( G.m )
    π = [i for i ∈ n:-1:1];
    return  da_clean_inner( G, α, L, max_degree, f_prune_curr, π );
end

function  disk_ann_build_graph( m, α, L, max_degree )
    n = size( m )
    _G = generate_directed_random_graph( n, 2*max_degree )
    G = NNGraph( n, m, _G, "disk_ann" );
    #L = 20;
    
    da_clean( G, α, L, max_degree )
    da_clean( G, α, L, max_degree )
    
    return  G
end


function  test_nn_queries( m_i, m_q )
    T = TimerOutput();
    
    @assert( size( m_i, 1 ) == size( m_q, 1 ) );
    d::Int64 = size( m_i, 1 );
    n::Int64 = size( m_i, 2 );
    n_q::Int64 = size( m_q, 2 );
    
    println( "Dimension: ", d );
    println( "n        : ", n );
    println( "n_q      : ", n_q );
    
    PS = MPointsSpace( m_i );
    
    m = PermutMetric( PS );
    mp = PermutMetric( PS );

    println( "Computing greedy permutation" );
    @timeit T "Greedy Permutation" I, D = greedy_permutation_naive( PS, n )
    
    
    f_check_slow = false;
    if  ( f_check_slow )
        println( "Checking that the greedy DAG indeed works!" );
        check_greedy_dag_failure_slow( m_i, m_q, I, D, 1.01 );
    end

    #println( "Random graph: " );
    #@timeit T "RGraph" G_rand = nng_random_dag( MPointsSpace( m_i ), 10 );
    #description!( G_rand, "Random (10)" );

    α = 1.4
    L = 40
    max_degree = 10
    Graphs = Vector{NNGraph}()

    function  cleaning( G, clean::Int64, str, desc, f_rev = false )
        if  f_rev
            str = str * "⇐"
        end
        for i ∈ 1:clean
            G = deepcopy( G )
            desc = desc * "C"
            str = str * "C"
            println( "Cleaning..." )
            flush( stdout );
            if  f_rev
                @timeit T str da_clean_rev( G, α, L, max_degree, true )
            else
                @timeit T str da_clean( G, α, L, max_degree, true )
            end
            
            description!( G, desc );
            push!( Graphs, G );
        end
    end
    
    function  greedy_graph( bfactor, clean::Int64 = 0  )
        println( "Building greedy graph ", bfactor );
        str = "GGraph " * string( bfactor );
        @timeit T str G = nng_greedy_graph( m_i, I, D, bfactor );    
        desc_str = "Greedy (" * string( bfactor ) * ")"

        description!( G, desc_str );
        push!( Graphs, G );

        cleaning( G, clean, str, desc_str );
        
        println( "Done\n" );
        flush( stdout );            
    end
    
    function  inc_graph( bfactor, clean::Int64 = 0  )
        println( "Incremental graph construction (", bfactor, ")" );
        m_inc = PermutMetric(  MPointsSpace( m_i ) );
        str = "GInc " * string( bfactor )
        @timeit T str  G, _ = nng_build_and_permut!( m_inc, bfactor );
        description!( G, str );
        push!( Graphs, G );
        flush( stdout );
        cleaning( G, clean, str, str );
    end

    function  inc_prune_graph( bfactor, clean::Int64 = 0, f_rev = false,
    f_store_first = true )
        println( "Incremental prune graph construction (", bfactor, ")" );
        m_inc = PermutMetric(  MPointsSpace( m_i ) );
        str = "GInc P " * string( bfactor )
        @timeit T str  G, _ = nng_build_and_permut!(
            m_inc, bfactor, true, α, 8*max_degree );
        description!( G, str );
        if  f_store_first 
            push!( Graphs, G );
        end
        flush( stdout );
        cleaning( G, clean, str, str, f_rev );
    end

    ####################################################################
    ####################################################################
    #greedy_graph( 1.1 );
    greedy_graph( 1.2 );
    greedy_graph( 2.0, 1 );
                
    #inc_graph( 1.3 );
    inc_graph( 1.6, 1 );
    inc_graph( 2.0, 1 );
    #inc_graph( 3.0, 2 );
        
    inc_prune_graph( 1.6, 1 );
    #inc_prune_graph( 2.0, 1, false );
    inc_prune_graph( 2.0, 1, true, true );


       ##########################################################################
    @timeit T "DiskAnn" G_disk_ann = disk_ann_build_graph( MPointsSpace( m_i ),
                                            α, L, max_degree )
    description!( G_disk_ann, "DiskAnn(" * string( α ) * "," * string( L )
                * "," * string(max_degree) );
    push!( Graphs, G_disk_ann );
    flush( stdout );

    #############################################################
    QS = MPointsSpace( m_q );
    lens = [1,2,4,8,16,20, 40, 80, 160, 320, 1000 ];

    function  query_testing() 
        println( "Query testing starting..." );
        #acc = zeros( 0.0, 2 );
        n_g = length( Graphs );
        println( typeof( n_g ));
        acc = zeros( n_g, 2 );
        for i ∈ 1:n_q
            ℓ = nn_search_brute_force( m_i, m_q[ :, i] )
            println( "Query ", i, " (", ℓ, "):" );

            rec = zeros( Float64, 2 );
            line = "FAIL";
            for g ∈ 1:length(Graphs)
                G = Graphs[ g ];
                f_first = true
                for  k ∈ lens
                    str, f_exact, perf = nng_nn_search_print( G, QS.m[ :, i ], k, ℓ )
                    if  f_first
                        f_first = false
                        rec = perf
                    end
                    if  f_exact 
                        line = str
                        rec = perf;
                        break
                    end
                end
                acc[g,:] += rec
                @printf( "---- %-25s %s\n", G.desc, line );
                #                println( "" );
                flush( stdout );
            end
        end

        return  acc
    end

    acc = query_testing()
    
    println( "\nn: ", n );
    println( "----------------------------------------------------" );
    flush( stdout );
    for  g ∈ 1:length(Graphs)
        G = Graphs[ g ]
        @printf( "%-18s #e: %7d avg_d: %7.2f avg_k: %6.2f  avg_c: %g\n", G.desc,
            ne( G.G ), ne(G.G)/n,
            acc[g,1] / n_q, # average k
            acc[g,2] / n_q  # average cost
        );
    end
    println( "----------------------------------------------------" );
    flush( stdout );

    show( T, allocations=false, compact = true )
    println( "" );

    #println( "GInc 1.6 #Edges before ", m_inc_1_6_b );
    #println( "GInc 1.6 #Edges after  ", m_inc_1_6_a );
    flush( stdout );
    
    return 0;
end



function  (@main)(args)
    #m_i, m_q = input_random( 100, 10, 2 )
    #m_i, m_q = input_random( 1000, 50, 8 )
    m_i, m_q = input_sift_small()# # 10000, 1000, 8 );
    test_nn_queries( m_i, m_q )
    return  0
end


