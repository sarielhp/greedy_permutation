#! /bin/env julial

push!(LOAD_PATH, pwd()*"/src/cg/")
push!(LOAD_PATH, pwd()*"/src/" )

using Graphs
using Distances
using DataStructures
using Printf, Format
using Random
using Accessors
using DiGraphRW
using JLD2
#using PlotGraphviz

#using SimpleWeightedGraphs

using TimerOutputs

using MetricSpace
using MaxDegreeDiGraph

include( "utils.jl" )
include( "read_fbin.jl" )

function  EmptyDiGraph( n::Int, ub_max_deg::Int )
    return  DiGraph( n )
end


function  EmptyMaxDegGraph( n::Int, ub_max_deg::Int )
    return   MaxDegDiGraph(n, ub_max_deg)
end

###################################################################

@kwdef mutable struct NEnv
    α::Float64 = 0.0
    β::Float64 = 0.0
    L::Int = 40
    R::Int64 = 4     # Max-degree
    R_clean::Int64 = 8     # Max-degree used as threshold for cleaning

    dim::Int = 0
    n::Int = 0
    n_q::Int = 0

    f_load = true  # Load graph from file if exists
    f_save = true  # Save graph to file
    data_dir::String = "data/"
    input_name::String = "undef" 
    saves_dir::String = "saves/"
    
    ub_max_degree::Int = 12   # Max-degree upper bound
    T::Union{TimerOutput,Nothing} = TimerOutput()
    #GG::Vector{NNGraph} = Vector{NNGraph}()
    
    ring_factor::Float64 = 1.1
    
    out_file_count::Int64 = 0
    init_graph::Function = EmptyDiGraph
end

#mutable struct NNGraph{FMS} where {FMS <: AbstractFMetricSpace}
mutable struct NNGraph{FMS <: AbstractFMetricSpace, DGraph}
    env::NEnv 
    n::Int
    m::FMS
    G::DGraph
    desc::String
    filename::String
end

function  NNGraph( env::NEnv, _m::FMS, _n::Int, G::DGraph ) where{FMS, DGraph}
    #    _n = size( _m )
    #DiGraph( _n )
    return NNGraph{FMS,DGraph}( env, _n, _m, G, "", "" )
end        

function  description!( G::NNGraphF, _desc::String ) where {NNGraphF}
    G.desc = _desc
end

"""
    nng_random_dag!(_G::NNGraph{FMS}, d::Int = 10) where {FMS}

Generates a random directed acyclic graph (DAG) for the given nearest neighbor graph (`NNGraph`).

# Arguments
- `_G::NNGraph{FMS}`: The nearest neighbor graph object, where `FMS` is a finite metric space.
- `d::Int`: The maximum number of outgoing edges (degree) for each vertex. Defaults to `10`.

# Behavior
- For each vertex `i` in the graph, the function randomly selects up to `d` vertices from the set of vertices with indices greater than `i` (to ensure the graph remains acyclic).
- The selected vertices are added as outgoing edges from vertex `i`.
- Duplicate edges are removed using `unique!`.
"""
function nng_random_dag!( _G::NNGraph{FMS}, n::Int, d::Int = 10 ) where {FMS}
    G = _G.G
    for  i ∈ 1:(n-1)
        dst = rand( (i+1):n, d )
        unique!( dst )
        for j ∈ dst
            add_edge!( G, i, j )
            @assert( outdegree( G, u ) <= _G.env.ub_max_degree )
        end
    end
end

function nng_random_dag( env, m::FMS, degree::Int, _G ) where {FMS}
    G_rand = NNGraph( env, m, size( m ), env.init_graph( size(m), env.ub_max_degree ) )
    nng_random_dag!( G_rand, size( m ), degree )
    return  G_rand
end


function find_k_lowest_f(G, u::Int, k::Int, f )
    @assert( k > 0 )
    
    cost::Int = 0

    @inline function  eval_f( z )
        cost += 1
        return f( z )
    end
    
    #println( "find_k_lowest_f" )
    # Use a PriorityQueue to store vertices to visit, prioritized by their f value.
    # The lowest f value has the highest priority.
    pq     = PriorityQueue{Int, Float64}()
    result = PriorityQueue{Int, Float64}(Base.Reverse)
#    pw 
    u_val = eval_f( u )

    enqueue!( pq, u, u_val )
    enqueue!( result, u, u_val )
    
    #println( "u_val: ", u_val )
    
    # A set to keep track of visited vertices to avoid cycles and redundant processing.
    visited = Set{Int}()
    push!(visited, u)
    
    f_threshold = false
    threshold = -10.0

    # Loop until the PriorityQueue is empty or we have found k vertices.
    while !isempty(pq)
        # Dequeue the vertex with the current lowest f value.
        #println( "before dequeue!" )
        v, v_val = peek( pq )
        dequeue!(pq)

        ℓ = length( result )
        if  ( ℓ >= k )
            f_threshold = true
            while  ( length( result ) > k )
                dequeue!( result )
            end
            _, threshold = peek( result )
            #println( "threshold: ", threshold )
            # Maybe too aggressive?
            if  ( v_val > threshold )
                break
            end
        end

        # Put it into the results...
        result[ v ] = v_val

        N = outneighbors(G, v)

        for  o ∈ N
            if  ( o ∈ visited )
                continue
            end

            push!( visited, o )

            o_val = eval_f( o )
            # @assert( o_val > 0.0 )
            if  ( f_threshold   &&  ( threshold < o_val ) )
                continue
            end
            
            enqueue!( pq, o, o_val )
            enqueue!( result, o, o_val )
        end
    end

    cl = collect( result )
    rs = [cl[i]  for i ∈ length(cl):-1:1]

    return   rs, cost, visited
end


TVerHopVal = Tuple{Int,Int,Float64}

function  dump_visited( env, _V::Vector{TVerHopVal} )
    V = sort( _V, by = p -> p[ 3 ] )
    fname = find_first_available_file( env, "out/" )

    open( fname, "w") do io
        # Now, we iterate through each tuple in the sorted vector.
        println(io," Vertex, Hop, distance\n" )
        for (v,h,d) in V
            @printf(io, "%12d, %12d, %12g\n", v, h , d)
        end
        flush( io )
    end    
end



# Importantly, the next struct in not mutable...

struct  QEntry
    vertex::Int
    hop_dist::Int
    v_order::Int  # The order in which the point was visited
    prev::Int     # previous vertex in the BFS
    dist::Float64    
end


function  extract_path( v, vals_f, prev, path )
    while v > 0
        v_val = vals_f[ v ]
        push!( path, (v, v_val) )
        if  v == 1
            v = 0
        else
            v = prev[ v ]        
        end
    end
    return  path
end

"""
    find_k_lowest_f_greedy(env, G, u::Int, k::Int, f)

Finds the `k` vertices in the directed graph `G` (of type `DiGraph`) starting from vertex `u` that have the lowest values according to the function `f`.

# Arguments
- `G::DiGraph`: The directed graph to search.
- `u::Int`: The starting vertex.
- `k::Int`: The number of vertices with the lowest `f` values to find.
- `f`: A function that takes a vertex index and returns a `Float64` value.

Returns
-------
 A tuple `(rs, cost, visited, visited_out, path)` where:
   - `rs`: A vector of the `k` vertices with the lowest `f` values, in order
           from lowest to highest.
           Each element is a tuple `(vertex_id, (hop_distance, f_val))`.
   - `cost`: The number of times the function `f` was called (i.e., the number of value computations).
   - `visited`: A set of vertices that were visited during the search.
   - `visited_out`: A vector of tuples `(vertex_id, hop_distance, f_val)` representing the order in which vertices were visited.
   - `path`: A vector of tuples `(vertex_id, f_val)` representing the path from the starting vertex to the 
      closet vertex found.

# Notes
- The function uses a priority queue to efficiently find the `k` lowest `f` values.
- Memoization is used to avoid redundant calls to `f`.
- The function also tracks the path and hop distance, but only the vertices, cost, and visited set are returned.
- The function assumes that vertex indices are integers and that `f` is defined for all relevant vertices.

# Remarks

- A subtle issue is that a vertex that is not explicitly explored by
  the algorithm (i.e., dequeued), is useless as far as using it for
  "shortcutting" the graph. Thus, the visited set does not include this set.

- Note that the code is not optimized for speed (it should not be too
  bad, but still). An optimized implementation would probably skip
  QEntry and use direct relevant values instead. 

- the visit_order number of a vertex is determined only when it is
  being dequeued, and only then it is being pushed to
  visited_out. This way, more sophisticated (and hopefully more
  efficient) pruning might be possible.

"""
function find_k_lowest_f_greedy(env, G, u::Int, k::Int, f; f_path = false )
    @assert( k > 0 )
    
    # Use a PriorityQueue to store vertices to visit, prioritized by their f value.
    # The lowest f value has the highest priority.
    pq = PriorityQueue{Int, QEntry}(Base.Order.By(p -> p.dist))
    result = PriorityQueue{Int, QEntry}(Base.Order.By(p -> -p.dist))
    
    leftover = Vector{QEntry}()  # (vertex, prev_vertex, hop_dist) 
    visited = Set{Int}()
    queued = Set{Int}()
    visited_out = Vector{QEntry}()
    prev = Dict{Int,Int}()
    vals_f = Dict{Int,Float64}()
    cost::Int = 0  # The number of times f was called...
    f_threshold = false
    threshold = -10.0

    # F with memoization...
    @inline function  eval_f( y )
        if  haskey( vals_f, y )
            return  vals_f[ y ]
        end
        cost += 1
        val = f( y )
        vals_f[ y ] = val
    end
        
    function  queue_it( u, ent, prev_u )
        if  u ∈ queued
            return
        end
        prev[ u ] = prev_u
        push!( queued, u )
        enqueue!( pq, u, ent )
        enqueue!( result, u, ent )
    end

    function  leftover_to_queue()
        while  !isempty( leftover )
            #z, z_val = peek( leftover )
            z_ent = pop!( leftover )
            z = z_ent.vertex
            if  ( z ∈ visited  ||  z ∈ queued )
                continue
            end
            z_val = eval_f( z )
            @assert( z_val == z_ent.dist )
            if  ( f_threshold   &&  ( threshold < z_val ) )
                continue
            end
            
            queue_it( z, z_ent, z_ent.prev )
        end
    end

    # Threshold for greedy jump. This constant maybe should be optimized?
    δ = 0.9

    #####################################################################
    ### The real work starts... #########################################
    #####################################################################
    u_val = eval_f( u )
    u_ent = QEntry( u, 1, 1, u, u_val )
    queue_it( u, u_ent, u )
    curr = u_val
    
    visit_count = 0
    # Loop until the priority queue is empty or we have found k vertices.
    while ( ( !isempty(pq) )  ||  ( !isempty( leftover ) ) )
        if isempty( pq )
            leftover_to_queue()
            if  isempty( pq )
                break
            end
        end
        
        # Dequeue the vertex with the current lowest f value.
        visit_count += 1

        v::Int, v_ent::QEntry = peek( pq )
        @reset v_ent.v_order = visit_count

        push!( visited_out, v_ent )
        push!( visited, v )

        v_hop, v_val = v_ent.hop_dist, v_ent.dist
        dequeue!(pq)

        if  v_val < curr
            curr = v_val
        end

        #-------------------------------------------------------------------------
        # Trim the result queue so that it does not contains more than k results
        #-------------------------------------------------------------------------
        if  ( length( result ) >= k )
            f_threshold = true
            while  ( length( result ) > k )
                dequeue!( result )
            end
            _, peek_ent = peek( result )
            threshold = peek_ent.dist
            if  ( v_val > threshold )
                break
            end
        end

        # Put it into the results...

        ### Handle the outgoing edges from v
        N = outneighbors(G, v)
        f_copy = false
        for  o ∈ N
            if  ( o ∈ visited  ||  o ∈ queued )
                continue
            end
            # Not that we *must* evaluate f on o, so we might as well do it now...
            o_val = eval_f( o )
            o_ent = QEntry( o,v_hop+1, -1, v, o_val )
            if  f_copy
                push!( leftover, o_ent ) 
                continue
            end
            
            if  ( f_threshold   &&  ( threshold < o_val ) )
                continue
            end

            queue_it( o, o_ent, v )
            
            # Greedily jump if the improvement is significant...
            if  ( o_val < δ * curr )
                f_copy = true
                curr = o_val
            end
        end
    end

    cl = collect( result )    
    rs::Vector{QEntry} = [cl[i][2]  for i ∈ length(cl):-1:1]

    path = Vector{Tuple{Int,Float64}}()
    if  f_path
        extract_path( rs[ 1 ].vertex, vals_f, prev, path )
    end
    
    return   rs, cost, visited, visited_out, path
end


function  nn_add_edges!( env, G, m, i, r_i, factor, R )
    function  dist_to_i( j )
        return  dist( m, i, j )
    end

    out = Vector{Int}()
    result, _ = find_k_lowest_f_greedy(env, G, 1, env.L, dist_to_i )
    forced::Int = 1 
    for ent ∈ result
        #(u, (hop,dista))
        u = ent.vertex
        #hop = ent.hop_dist
        dista = ent.dist
        #QEntry
        if (dista < factor * r_i ) ||  ( forced > 0 )
            forced -= 1
            if  ( outdegree( G, u ) < env.ub_max_degree )
                add_edge!(G, u, i)
                push!( out, u )
            end
        end
    end

    return  out
end


function vertex_prune!( env, _G, v )
    U = Set{Int}( outneighbors( _G.G, v ) )
    robust_prune_vertex!( _G, v, U, env.α, env.R )
end

function  add_edge_and_prune!( env, _G, u, v )
    if  has_edge( _G.G, u, v )
        return
    end

    if  ( outdegree( _G.G, u ) >= env.R_clean )
        vertex_prune!( env, _G, u )
    end
    
    @assert( outdegree( _G.G, u ) < env.ub_max_degree )
    
    add_edge!( _G.G, u, v )
end


function   prune_candidates!( C::Vector{QEntry}, i_del, is_delete::Function )
    ent = C[ i_del ]

    delete_fast!( C, i_del )
    for  i ∈ length(C):-1:1
        if is_delete( ent.vertex, C[ i ].vertex )
            delete_fast!( C, i )
        end
    end

    return  C
end

"""
nn_add_edges_prune

Compute all the vertices that in distance at most r_i * factor from v,
and then prunes them to make sure there not too many of them, and
create edges from the survivors of the pruning to v.
"""
function  nn_add_edges_prune( env, G, m, v, r_i, factor, R )
    function  dist_to_v( j )
        return  dist( m, v, j )
    end

    out = Vector{Int}()
    results, _ = find_k_lowest_f_greedy(env, G.G, 1, env.L, dist_to_v )
    r_ub = r_i * factor

    # get only the vertices that are not v, but are not too far away
    C = filter( item -> ( 0.0 < item.dist <= r_ub ), results )
    out = Vector{Int}()
    while  true
        if  ( isempty( C ) )
            break
        end

        i_min::Int = argmin( item.dist for item ∈ C )
        r_min = C[ i_min ].dist
        r_max = r_min * env.ring_factor
        # indices contains all the entries of C that are "competitive" with the minimum.
        #indices = [i for i ∈ 1:length( C ) if C[ i ].dist <= r_max ]
        indices = findall( x -> ( x.dist <= r_max ), C)
        
        # The point with the minimum vertex visiting number is the
        #best one to shortcut to...
        ii_min = argmin( C[ i ].v_order for i ∈ indices )

        u_ind = indices[ ii_min ]
        u_ent = C[ u_ind ]
        u = u_ent.vertex
        push!( out, u )

        # Then we do pruning according to the Apollonius ball as done by DiskAnn... 
        # v -> u -> g
        function  is_delete( u, g )
            return   ( ( env.α * dist( m, u, g ) ) <= dist( m, v, g ) )
        end

        prune_candidates!( C, u_ind, is_delete )
    end

    for  x ∈ out
        add_edge_and_prune!( env, G, x, v )
    end
        
    return  out
end



function nng_greedy_dag!(_G::NNGraph{FMS},
    R::Vector{Float64},
    n::Int,
    factor::Float64=1.1
) where {FMS}
    @printf( "n    : ", format( n, commas=true ) )
    #@printf( "n_q  : %10'd\n", n_q)
    G = _G.G
    m = _G.m
    k = 200
    @assert(length(R) == n, "R must have the same length .")
    for i ∈ 2:n
        result, _ = find_k_lowest_f(G, 1, k, j -> dist( m, i, j ) )
        edges_added = 0
        forced = 2
        count = 0
        for (u, dist) ∈ result
            count += 1
            #f_add = ( count == 1 )  ||  
            # the i-1 is important in the following line! It is the true distance.
            if ( (dist < factor * R[ i - 1 ])  ||  ( forced > 0 ) )
                edges_added += 1
                forced -= 1
                if  ( outdegree( G, u ) < _G.env.ub_max_degree )
                    add_edge!(G, u, i)
                end
            end
        end

        if  ( edges_added == 0 )
            dst = dist( m, 1, i )
            for j = 1:i-1
                dst = min( dst, dist( m, i, j) )
            end
            println( "No edges added for : ", i )
            println( result )
            println( "Real dist: ", dst )
            println( R[ i - 1 ] )
            println( R[ i ] )
            exit( -1 )
        end
    end
end


function  nng_greedy_graph( env, n, m_i, I, D, factor )
    PS = MPointsSpace( m_i, n )
    m = PermutMetric( PS, I )
    G = NNGraph( env, m, n, env.init_graph( n, env.ub_max_degree ) )

    nng_greedy_dag!( G, D, n, factor )
    return  G
end


function  nng_nn_search_all( env, _G::NNGraph{FMS}, 
                             q::Int,
                                 L::Int = 20
                                ) where {FMS}
    result, _, visited, visited_out,_ =
         find_k_lowest_f_greedy( env, _G.G, 1, L, j -> dist( _G.m, j, q ) )

    return  result, visited, visited_out
end


"""
Returns only the distance to the nearest neighbor.
"""
function  nng_nn_search_reg( env, _G::NNGraph{FMS}, 
                                 q::Int,
                                 k::Int = 20
                                ) where {FMS}
    result, _ = nng_nn_search_all( env, _G, q, k )

    return result[ 1 ].dist
end

function  nng_nn_search( env, _G::NNGraph{FMS}, 
                                 q_real,
                                 k::Int = 20
                                ) where {FMS}
    result, cost = find_k_lowest_f( _G.G, 1, k, j -> dist_real( _G.m, j, q_real) )
    
    return  result[ 1 ]
end

function  shortcut_path( path::Vector{Tuple{Int,Float64}},
                        β::Float64 = 1.1 )
    opath = Vector{Tuple{Int,Float64}}()
    if length( path ) < 3
        return  opath
    end
    i = length( path ) 
    curr = path[i][2] 
    push!( opath, path[ i ] )
    f_prev = true;
    i -= 1
    while  i > 0
        v,v_val = path[ i ]
        if  (v_val < curr/β)  ||  ( i == 1 )
            if  f_prev
                f_prev = false
            else
                push!( opath, (v, v_val ) )
                f_pref = true;
                curr = v_val
            end
        else
            f_prev = false;
        end
        i -= 1
    end
    return  opath
end


function  nng_nn_search_print( env, _G::NNGraphFMS, 
                               q_real,
                               k::Int,
                               ℓ::Float64
                                ) where {NNGraphFMS}

    result_g, cost_g, _, _, path =
         find_k_lowest_f_greedy( env, _G.G, 1, k,
                                 j -> dist_real( _G.m, j, q_real) )

    dist_g = result_g[1].dist
    f_exact_g = ( dist_g == ℓ )
    s_dist_g = f_exact_g ? "Exact" : @sprintf( "%12g", dist_g )
    
    str = @sprintf( "k: %4d: cost: %7d   d: %s", k, cost_g,
                    s_dist_g  )
    return str, f_exact_g, [Float64(k), Float64( cost_g ) ]
end


function is_power_of_two(n::Integer)
    (n > 0)  &&  ((n & (n - 1)) == 0)
end

function is_power_of_four(n::Integer)
    if   ( ! is_power_of_two( n ) )
        return  false;
    end
    return is_power_of_two( n ÷ 2 )  
end

"""
Build the nearest-neighbor search graph together with its permutation.
The permutation is encoded in the permutation metric (i.e., m(i) is
the i point in the greedy permutation).
"""
function nng_build_and_permut!(
    env::NEnv,
    m::PermutMetric{FMS},
    factor::Float64 = 1.1,
    f_prune = false,
    f_shortcut = false,
    f_shortcut_clean = false
) where {FMS}
    n = env.n
    @assert( size( m ) == n ) 
    G = NNGraph( env, m, n, env.init_graph( n, env.ub_max_degree ) )

    R = env.R
    R = fill( -1.0, n )
    R[ 1 ] = 0.0

    # queue + radii 
    qadii = PriorityQueue{Int, Float64}(Base.Reverse)
    for i ∈ 2:n
        R[ i ] = dist( m, 1, i )  # i == π[ i ] so redundant to use π 
        enqueue!( qadii, i, R[ i ] )
    end

    println( "n    : ", format(n, commas=true ) )
    i::Int = 2
    c_count = 0
    i_iter::Int = 0
    handled::Int = 0
    while  ( !isempty( qadii ) )
        i_iter += 1
        curr, r_curr = peek( qadii )
        new_rad = nng_nn_search_reg( env, G, curr, env.L )
        #println( "Okay!" )
        if  ( new_rad < 0.99 * r_curr )
            qadii[ curr ] = r_curr = new_rad

            # Maybe it did not change after all
            ncurr, _ = peek( qadii )
            if  ( ncurr != curr )
                c_count += 1
                #                println( "CONTINUE!" )
                continue
            end
        end

        # Okay, the point curr is the next point in the greedy permutation...
        swap!( m, i, curr )

        # A bit of shenanigan with the queue
        r_i = qadii[ i ]
        qadii[ curr ] = r_i
        dequeue!( qadii, i )

        handled += 1
        
        # Now we need to add the edges to the DAG...
        N = nn_add_edges!( env, G.G, m, i, r_i, factor, env.R )

        if   f_shortcut
            clean_shortcut_vertex( env, G, i )
        end
        
        # Pruning is done only after we constructed at least half the graph...
        if   f_prune  ||  f_shortcut #( f_prune  &&  ( handled > (n/2) ) )
            for v ∈ N
                if  outdegree( G.G, v ) > 4*env.R
                    U = Set{Int}( outneighbors( G.G, v ) )
                    robust_prune_vertex!( G, v, U, env.α, 2*env.R )
                end
            end                    
        end
        R[ i ] = r_curr

        if  f_shortcut_clean  &&  ( ( i > 128 )  &&  is_power_of_four( i ) )
            clean_shortcut( env, G, i )
        end
        
        i += 1
    end

    println( "c_count: ", c_count, " / ", i_iter )
    return  G, R
end


function  queue_points( m )
    n = size( m )
    println( "n    : ", format(n, commas=true ) )
    R = fill( -1.0, n )
    qadii = PriorityQueue{Int, Float64}(Base.Reverse)
    for i ∈ 2:n
        R[ i ] = dist( m, 1, i )  # i == π[ i ] so redundant to use π 
        enqueue!( qadii, i, R[ i ] )
    end
    R[ 1 ] = maximum( R[2:n] )

    return  qadii, R 
end

function nng_build_prune!( env::NEnv, m::PermutMetric{FMS}, factor::Float64 = 1.1
                           ) where {FMS}
    n = size( m )
    G = NNGraph( env, m, n, env.init_graph( n, env.ub_max_degree ) )

    qadii, R = queue_points( m )

    i::Int = 2
    c_count = 0
    i_iter::Int = 0
    handled::Int = 0
    while  ( !isempty( qadii ) )
        i_iter += 1
        curr, r_curr = peek( qadii )
        new_rad = nng_nn_search_reg( env, G, curr, env.L )

        if  ( new_rad < 0.99 * r_curr )
            qadii[ curr ] = r_curr = new_rad

            # Maybe it did not change after all
            ncurr, _ = peek( qadii )
            if  ( ncurr != curr )
                c_count += 1
                continue
            end
        end

        # Okay, the point curr is the next point in the greedy permutation...
        swap!( m, i, curr )

        # A bit of shenanigan with the queue
        r_i = qadii[ i ]
        qadii[ curr ] = r_i
        dequeue!( qadii, i )

        handled += 1
        
        # Now we need to add the edges to the DAG...
        nn_add_edges_prune( env, G, m, i, r_i, factor, env.R )

        R[ i ] = r_curr
        
        i += 1
    end

    return  G, R
end


function  input_deep1b( n::Int = 0 )
    basedir = "data/deep1b/"
    m_i = read_fbin_n( basedir * "base.10M.fbin", n )
    m_q = read_fbin_n( basedir * "query.public.10K.fbin" )
    return  m_i, m_q
end


function  input_sift_small( env )
    basedir = "data/sift/"
    env.input_name = "sift_small";

    m_i = read_fvecs( basedir * "small_base.fvecs" )
    m_q = read_fvecs( basedir * "small_query.fvecs" )
    return  m_i, m_q
end

function  input_sift( env, n::Int = 0 )
    basedir = "data/sift/"
    env.input_name = "sift";
    m_i = read_fvecs( basedir * "base.fvecs" )
    if  ( n > 0 )
        m_i = m_i[ :, 1:n ]
    end

    env.input_name = "sift";
    m_q = read_fvecs( basedir * "query.fvecs" )
        
    return  m_i, m_q
end

compute_input_filename( env ) = env.data_dir * "gen/" * env.input_name * ".jld2"

function  load_input( env )
    fln = compute_input_filename( env )
    if  isfile( fln )
        @load fln m_i m_q
        println( "Input loaded from: ", fln )
        flush( stdout )
        return  true, m_i, m_q
    end
    return  false, Matrix{Float64}(), Matrix{Float64}()
end

function  save_input( env, m_i, m_q )
    fln = compute_input_filename( env )
    create_dir_from_file( fln )
    @save fln m_i m_q
end


function  input_random( env, n, n_q, d )
    env.input_name = "rand_n" * string(n)* "_q"*string(n_q)*"_d"*string(d);
    f_load, m_i, m_q = load_input( env )
    if  f_load
        println( "Loaded input..." )
        return  m_i, m_q
    end
    m_i_, m_q_ = rand( d, n ), rand( d, n_q )
    if   env.f_save
        save_input( env, m_i_, m_q_ )
    end

    return  m_i, m_q
end


function  nn_search_brute_force( n, m_i, q_real )::Float64
    dst::Float64 = euclidean( m_i[ :, 1 ], q_real )
    for  j ∈ 2:n
        new_dst = euclidean( m_i[ :, j ], q_real )
        if  ( new_dst < dst )
            dst = new_dst
            if  ( dst == 0.0 )
                println( "DST: ", euclidean( m_i[ :, j ], q_real )  )
                println( "j: ", j )
            end
        end
    end

    return  dst
end

"""
    Check that the greedy graph has strong connectivity and all the nearest-neighbors are reachable from the first vertex.
"""
function  check_greedy_dag_failure_slow( env, m_i, m_q, I, D, factor )
    d::Int = env.dim
    n_q::Int = env.n_q
    
    PS = MPointsSpace( m_i, env.n )
    m = PermutMetric( PS, I )
    GB     = NNGraph( m, n, env )

    nng_greedy_dag!( GB, D, n, factor )    
    for i ∈ 1:n_q
        real_dist = nn_search_brute_force( n, m_i, m_q[ :, i] )
            
        dist = nng_nn_search( env, GB, m_q[ :, i ], n )[ 2 ]

        if  ( dist != real_dist )
            println( "greedy DAG failed for ", factor )
            println( "dist      : ", dist )
            println( "real_dist : ", real_dist )
            println( "Query: ", m_q[ :, i ] )

            G = DiGraph( GB.G )
            dot = plot_graphviz( G )
            write_dot_file( G, "test.dot" )

            for  e ∈ edges( GB.G )
                println( e )
            end
            println( D )
            exit( -1 )
        end
    end
    println( "Greedy DAG passed for: ", factor )
end

"""
Given a set of vertices V in the metric of G, the following function
robustly prune it as described in the Disk-Ann paper, 
iterative adding the closest point to the out-set, while removing its Apollonius ball from V. 
"""
function  robust_prune( G::NNGraphFMS, p, V::Set{Int},
                        α::Float64, max_size ) where {NNGraphFMS}
    pq     = PriorityQueue{Int, Float64}()

    N = deepcopy( V ) #union( Set{Int}( ), V )
    for x ∈ outneighbors(G.G, p)
        push!( N, x )
    end
        
    delete!( N, p )
    
    for  o ∈ N
        enqueue!( pq, o, dist( G.m, p, o ) )
    end

    out = Set{Int}()
    while  ( ! isempty( pq ) )
        v, v_val = peek( pq )
        dequeue!(pq)
        push!( out, v )
        if  ( length( out ) >= max_size )
            break
        end
        for xp ∈ pq
            x = xp[1]
            if   α * dist( G.m, v, x ) <= dist( G.m, p, x )
                delete!( pq, x )
            end
        end
    end
    
    return out
end

function  robust_prune_hop!( env, G, p::Int, res::Vector{QEntry} )
    α, R = env.α, env.R

    out = Set{Int}()
    while  ( ! isempty( res ) )
        i = argmin( ent.dist  for ent ∈ res )

        v_val = res[ i ].dist
        ub = v_val * env.ring_factor

        # get vertex with distance in same ring as minimum distance ring, and
        # minimal hop distance...
        res_filtered = filter(t -> ( v_val <= t.dist <= ub), res)
        #println( typeof( res_filtered ) )
        #println( length( res_filtered ) )
        #dump( res_filtered )
        
        ihop = argmin( x.hop_dist  for x ∈ res_filtered )
#        res_filtered, by =   [ 2 ])

        ent = res_filtered[ ihop ]
        v, v_val = ent.vertex, ent.dist
        #v, v_val = pq_filtered[ ihop ]        
        #v, v_hop, v_val = find_minimal_vertex( pq )
        
        push!( out, v )
        if  ( length( out ) >= R )
            break
        end
        for i ∈ length( res ):-1:1
            ent = res[ i ]
            x = ent.vertex
            if  ( ( ent.vertex == v )
                  ||  ( α * dist( G.m, v, x ) <= dist( G.m, p, x ) ) )
                delete_fast!( res, i )
            end
        end
    end
    
    return  out
end


function  robust_prune_vertex!( G::NNGraphFMS, p, V::Set{Int},
                           α, R ) where {NNGraphFMS}
    out =  robust_prune( G, p, V, α, R )
    
    neighbors_to_delete = Set{Int}( collect( outneighbors(G.G, p) ) )
    del_list = setdiff( neighbors_to_delete, out )
    add_list = setdiff( out, neighbors_to_delete )
    
    for t in del_list
        rem_edge!( G.G, p, t)
    end
    for t in add_list
        add_edge!( G.G, p, t)
        @assert( outdegree( G.G, p ) <= G.env.ub_max_degree )
    end
end


function random_perm_no_selfies( n )
    while  true
        p = randperm(n)

        f_good = true
        for i ∈ 1:n
            if  p[ i ] == i
                f_good = false
                break
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
        if  ( n > 100000 )
            print( "            round: ", i, "/", d, "    \r" )
            flush( stdout )
        end
        p = random_perm_no_selfies( n )
        for  i ∈ 1:n
            if  ! has_edge( G, i, p[ i ] )
                add_edge!( G, i, p[ i ] )
            end
        end
    end

    println( "" );
    
    return G
end

function  da_clean_inner( env, G, π::Vector{Int};
                          f_hop_clean = false )
    α, R, L = env.α, env.R, env.L
    RX = R + 10
    n = env.n

    rpv::Int = 0
    for i ∈ 1:n
        #if  ( i & 0x7ff ) == 0 
        #    @printf( "%10d   \r", i )
        #    flush( stdout )
        #end
#=        if  ( f_prune_curr  &&  ( outdegree( G.G, π[ i ] ) > R ) )
            U = Set{Int}( outneighbors( G.G, π[ i ] ) )
            robust_prune_vertex!( G, π[ i ], U, env.α, env.R )
        end =#
        #XXX
        _, cost, V, V_v_hop_val,_ =
           find_k_lowest_f_greedy( env, G.G, 1, L, j -> dist( G.m, j, π[i] ) )

        #_, V, V_v_hop_val = nng_nn_search_all( env, G, π[ i ], L )
        if  ( i & 0xfff ) == 0 
            println( "cost: ", cost, " size V: ", length( V ), " i: ", i,
                "/", format( n, commas=true ),  "  rpv: ", rpv )
        end
        flush( stdout ) 
        if  f_hop_clean 
            out = robust_prune_hop!( env, G, π[ i ], V_v_hop_val )
        else
            out = robust_prune( G, π[ i ], V, α, R )
        end
        for j ∈ out
            # π[ i ]???? Verify
            if  ( outdegree( G.G, j ) >= RX )
                U = Set{Int}( outneighbors( G.G, j ) )
                push!( U, π[ i ] )
                robust_prune_vertex!( G, j, U, α, R )
                rpv += 1
            else
                add_edge!( G.G, j, π[ i ] )
                @assert( outdegree( G.G, j ) <= env.ub_max_degree )
            end
        end
    end
    
    return  G
end

function  da_clean( env, G; f_hop_clean::Bool = false )
    n = size( G.m )
    π = randperm(n)
    return  da_clean_inner( env, G, π, f_hop_clean=f_hop_clean )
end

function  da_clean_rev( env, G, f_hop_prune::Bool = false )
    n = size( G.m )
    π = [i for i ∈ n:-1:1]
    return  da_clean_inner( env, G, π, f_hop_prune )
end


function  clean_shortcut_vertex( env, G, v )
     _, R, _, β = env.α, env.R, env.L, env.β 

    if  ( outdegree( G.G, v ) > R )
        U = Set{Int}( outneighbors( G.G, v ) )
        robust_prune_vertex!( G, v, U, env.α, env.R )
    end
    _, _, _, _, path =
        find_k_lowest_f_greedy( env, G.G, 1, env.L, j -> dist( G.m, j, v ),
                                f_path = true )
    opath = shortcut_path( path, β )
    
    for i ∈ 1:length( opath ) - 1
        x = opath[ i ][1]
        y = opath[ i + 1 ][1]
        
        #println( x, " -> ", y ); #XXX
        @assert( x < y )
        if  ( outdegree( G.G, x ) < env.ub_max_degree )
            add_edge!( G.G, x, y )
        end
        
        if  ( outdegree( G.G, x ) > R )
            U = Set{Int}( outneighbors( G.G, x ) )
            robust_prune_vertex!( G, x, U, env.α, env.R )
        end
    end
end


function  clean_shortcut( env, G, _n::Int = 0 )

    n = ( _n == 0 ) ? size( G.m ) : _n;
    
    π = randperm(n)
    #α, R, L, β = env.α, env.R, env.L, env.β 

    println( "Clean shortcut" );
    for i ∈ 1:n
        clean_shortcut_vertex( env, G, π[ i ] )
        #clean_shortcut_vertex( env, G, n+1-i )
    end
    
    return  G
end


function  disk_ann_build_graph( env, m; f_hop_clean = false )
    n = size( m )
    println( "      Random graph..." )
    flush( stdout )
    @timeit env.T "Rand Graph" _G = generate_directed_random_graph( n, env.R )
    G = NNGraph( env, n, m, _G, "disk_ann", "" )
    hop_clean_str = f_hop_clean ? "_H_" : "_"
    G.filename = env.saves_dir * "da" * hop_clean_str * env.input_name*"_n" *
                  string(n)*"_R"*string(env.R) * ".bin"

    if  env.f_load
        if  isfile( G.filename )
            G.G = graphRead( G.filename ) 
            return
        end
    end
    
    println( "      Clean 1..." )
    flush( stdout )
    @timeit env.T "Clean 1" da_clean( env, G; f_hop_clean=f_hop_clean )

    println( "      Clean 2..." )
    flush( stdout )
    @timeit env.T "Clean 2" da_clean( env, G; f_hop_clean=f_hop_clean )

    if  env.f_save
        create_dir_from_file( G.filename )
        graphWrite( G.filename, G.G )
        H = graphRead( G.filename )
    end
    
    return  G
end


function  query_testing_inner( env, GG, n, m_i, m_q ) 
    n_q::Int = env.n_q
    println( "Query testing starting..." )
    n_g = length( GG )

    QS = MPointsSpace( m_q )
    lens = [1,2,4,8,16,20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 160, 320, 1000 ]


    #println( typeof( n_g ))
    acc = zeros( n_g, 2 )
    for i ∈ 1:n_q
        ℓ = nn_search_brute_force( n, m_i, m_q[ :, i] )
        println( "Query ", i, " (", ℓ, "):" )
        
        line = "FAIL"
        for g ∈ 1:length( GG )
            G = GG[ g ]
            f_first = true
            rec = zeros( Float64, 2 )
            for  k ∈ lens
                str, f_exact, perf = nng_nn_search_print( env, G, QS.m[ :, i ], k, ℓ )
                if  f_first
                    f_first = false
                    rec = perf
                end
                if  f_exact 
                    line = str
                    rec = perf
                    break
                end
            end
            acc[g,:] += rec
            @printf( "---- %-25s %s\n", G.desc, line )
            #                println( "" )
            flush( stdout )
        end
    end
    
    return  acc
end


function  cleaning( env, GG, G, clean::Int, str, desc,
                        f_rev = false, f_shortcut = false )
    T = env.T
    @assert( nv( G.G ) == G.n )
    if  f_rev
        str = str * "⇐"
    end
    for i ∈ 1:clean
        G = deepcopy( G )
        desc = desc * "C"
        str = str * "C"
        println( "Cleaning..." )
        flush( stdout )
        if  f_shortcut
            @timeit T str clean_shortcut( env, G )
        else
            if  f_rev
                @timeit T str da_clean_rev( env, G, true )
            else
                @timeit T str da_clean( env, G, true )
            end
        end
        
        description!( G, desc )
        push!( GG, G )
    end
end

function  bstring( b::Bool )::String
    return  b ? "T" : "F"
end


function  greedy_permtuation( env, m_i, n, m_q )
    PS = MPointsSpace( m_i, n )
    println( "Computing greedy permutation" )
    flush( stdout )
    @timeit env.T "Greedy Permutation" I, D = greedy_permutation_naive( PS, n )
    println( "Computing greedy permutation done." )
    flush( stdout )
    
    
    f_check_slow = false
    if  ( f_check_slow )
        println( "Checking that the greedy DAG indeed works!" )
        check_greedy_dag_failure_slow( m_i, m_q, I, D, 1.01 )
    end
    return  I, D
end



function  greedy_graph( env, GG, bfactor, clean::Int = 0  )
    println( "Building greedy graph ", bfactor )
    flush( stdout );
    str = "GGraph " * string( bfactor )
    @timeit T str begin
        @timeit T "building" G = nng_greedy_graph( env, n, m_i, I, D, bfactor )
        desc_str = "Greedy (" * string( bfactor ) * ")"
        
        description!( G, desc_str )
        push!( GG, G )
        
        cleaning( env, GG, G, clean, str, desc_str )
        
        println( "Done\n" )
        flush( stdout )            
    end
end


function  disk_ann_compute( env, GG, m_i, n )
    println( "Computing DiskAnn..." )
    flush( stdout )
    @timeit env.T "DiskAnn" G_disk_ann =
               disk_ann_build_graph( env, MPointsSpace( m_i, n ) )
    println( "Computing DiskAnn done..." )
    
    description!( G_disk_ann, "DiskAnn(" * string( env.α )
                              * "," * string( env.L )
                              * "," * string(env.R) )
    push!( GG, G_disk_ann )
    flush( stdout )
end


function  disk_ann_hop_compute( env, GG, m_i, n )
    println( "Computing DiskAnnHop..." )
    flush( stdout )
    @timeit env.T "DiskAnnH" G_disk_ann_h =
           disk_ann_build_graph( env, MPointsSpace( m_i, n ); f_hop_clean=true )

    description!( G_disk_ann_h, "DiskAnnHOP(" * string( env.α )
                              * "," * string( env.L )
                * "," * string(env.R) )
    push!( GG, G_disk_ann_h )
    println( "     done..." )
    flush( stdout )
end

function  test_nn_queries( env, m_i, m_q )
    n = env.n
    n_q = env.n_q
    T = env.T
    
    @assert( size( m_i, 1 ) == size( m_q, 1 ) )
    d::Int = size( m_i, 1 )
    @assert( 0 < n <= size( m_i, 2 ) )
    
    println( "Dimension: ", d )
    println( "n    : ", format(n, commas=true ) )
    println( "n_Q  : ", format(n_q, commas=true ) )
    flush( stdout )
    
    GG = Vector{NNGraph}()
    
    function  inc_graph( bfactor, clean::Int = 0  )
        println( "Incremental graph construction (", bfactor, ")" )
        m_inc = PermutMetric(  MPointsSpace( m_i, n ) )
        str = "GInc " * string( bfactor )
        @timeit T str  G, _ = nng_build_and_permut!( env, m_inc, bfactor )
        description!( G, str )
        push!( GG, G )
        flush( stdout )
        cleaning( env, GG, G, clean, str, str )
    end

    function  inc_prune_graph( bfactor, clean::Int = 0, f_rev = false,
                               f_store_first = true )
        println( "Incremental prune graph construction (", bfactor, ")" )
        m_inc = PermutMetric( MPointsSpace( m_i, n ) )
        str = "GInc P " * string( bfactor )
        @timeit T str  begin
            @timeit T "building" G, _ = nng_build_and_permut!( env,
                m_inc, bfactor, true )
            description!( G, str )
            if  f_store_first 
                push!( GG, G )
            end
            flush( stdout )
            cleaning( env, GG, G, clean, str, str, f_rev )
        end
    end

    function  inc_graph_shortcut( bfactor, clean::Int = 0,
            f_store_first = true, f_shortcut_inc = false,
           f_shortcut_clean = false )
        println( "GSInc graph shortcut construction (", bfactor, ")" )
        m_inc = PermutMetric( MPointsSpace( m_i, n ) )
        str = "GSInc " * string( bfactor ) * "_" * string(clean) * "_" *
              bstring( f_store_first ) * bstring( f_shortcut_inc ) *
              bstring( f_shortcut_clean )
        if  f_shortcut_inc
            str = str * "[SC]"
        end
        if  f_shortcut_clean
            str = str * "<C>"
        end
        
        @timeit T str begin
            @timeit T "building" begin
                G, _ = nng_build_and_permut!( env,
                    m_inc, bfactor, false, f_shortcut_inc, f_shortcut_clean )
            end
            description!( G, str )
            if  f_store_first 
                push!( GG, G )
            end
            println( "Cleaning..." )
            flush( stdout )
            cleaning( env, GG, G, clean, str, str, false, true )
        end
    end

    function  pgs_inc_graph_shortcut( bfactor, clean::Int = 0, f_store_first = true )
        println( "PGSInc graph shortcut construction (", bfactor, ")" )
        m_inc = PermutMetric(  MPointsSpace( m_i, n ) )
        str = "PGSInc " * string( bfactor )
        @timeit T str begin
            @timeit T "Building" G, _ = nng_build_prune!( env, m_inc, bfactor )
            description!( G, str )
            if  f_store_first 
                push!( GG, G )
            end
            println( "Cleaning..." )
            flush( stdout )
            cleaning( env, GG, G, clean, str, str, false, true )
        end
    end

    ##########################################################################
    disk_ann_hop_compute( env, GG, m_i, n )
    disk_ann_compute( env, GG, m_i, n )

    # Generate and compute greedy permutation
    I, D = greedy_permtuation( env, m_i, n, m_q )

    #greedy_graph( 1.1 )
    #greedy_graph( env, 1.6, 1 ) # Complete waste of time
    #greedy_graph( env, 2.0, 1 )
                
    #inc_graph( 1.3 )
    #inc_graph( 1.6, 1 )
    #inc_graph( 2.0, 1 )
    #inc_graph( 3.0, 2 )
        
    #inc_prune_graph( 1.6, 1 )
    #inc_prune_graph( 2.0, 1, false )
    #inc_prune_graph( 2.0, 1, true, true )

    inc_graph_shortcut( 1.6, 1, true )
    inc_graph_shortcut( 1.6, 1, true, false, true )
    inc_graph_shortcut( 1.6, 3, true, true )
    pgs_inc_graph_shortcut( 1.6, 3, true )

#    inc_graph_shortcut( 2.0, 1, false )
    
    #############################################################

    acc = query_testing_inner( env, GG, n, m_i, m_q )
    
    println( "\nn: ", n )
    println( "----------------------------------------------------" )
    flush( stdout )
    for  g ∈ 1:length(GG)
        G = GG[ g ]
        @printf( "%-23s #e: %7d avg_d: %7.2f avg_k: %6.2f  avg_c: %g\n", G.desc,
            ne( G.G ), ne(G.G)/n,
            acc[g,1] / n_q, # average k
            acc[g,2] / n_q  # average cost
        )
    end
    println( "----------------------------------------------------" )
    flush( stdout )

    show( T, allocations=false, compact = true )
    println( "" )

    #println( "GInc 1.6 #Edges before ", m_inc_1_6_b )
    #println( "GInc 1.6 #Edges after  ", m_inc_1_6_a )
    flush( stdout )
    
    return 0
end

# On SIFT1M parameters DiskANN paper used:
# R=70
# L=125
# C=3000 (caching parameter?)
# α = 2

function  (@main)(args)
    println( "Starting..." )
    flush( stdout )
    env = NEnv()

    println( "Reading input" )
    flush( stdout )
    @timeit env.T "Reading input" begin
        m_i, m_q = input_random( env, 100, 10, 40 )
        #m_i, m_q = input_random( env, 2000, 50, 20 )
        #m_i, m_q = input_sift_small()# # 25000, 1000, 8 )
        #m_i, m_q = input_sift( env, 100 ) # # 1,000,000 points!
        #m_i, m_q = input_deep1b( 500_000 )
    end
    println( "Reading done" )
    flush( stdout )
    
    @assert( size( m_i, 1 ) == size( m_i, 1 ) )

    dim = size( m_i, 1 )
    n = size( m_i, 2 )

    dim_q = size( m_q, 1 )
    n_q = size( m_q, 2 )

    if  ( dim != dim_q )
        println( "Dimensions of data and queries, do not match!" )
        exit( -1 )
    end
    println( "dim : ", dim )
    println( "n    : ", format(n, commas=true ) )
    println( "n_Q  : ", format(n_q, commas=true ) )
    flush( stdout )

    #env.α = 1.4 
    env.α = 2.0
    env.β = 1.2 
    #env.R = 70   # "Max-degree" of the graph, more or less.

    # R is "Max-degree" of the graph, more or less.
#    env.R = min( round(Int, sqrt( n ) ), 70 )
    env.R = min( round(Int, sqrt( n ) ), 10 )
    env.L = round(Int, env.R + 40 )
    #env.L = 125
    #env.R = 10
    env.R_clean = round(Int64, 1.3 * env.R )
    env.ub_max_degree = env.R * 10
    env.init_graph = EmptyDiGraph
    env.n = n
    env.n_q = n_q
    env.dim = dim

    test_nn_queries( env, m_i, m_q )

    env.T = nothing 
    dump( env )

    println( "-----------------------------------------------------" )
    println( "dim  : ", dim )
    println( "n    : ", format(n, commas=true ) )
    println( "n_Q  : ", format(n_q, commas=true ) )
    
    #show( env )
    #println( env )
    
    return  0
end


