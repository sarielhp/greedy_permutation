#! /bin/env julia

using  BenchmarkTools
using  Graphs, Graphs.SimpleGraphs

# --- Example Usage (outside the module for clarity) ---

push!(LOAD_PATH, pwd()*"/src/" )

using MaxDegreeDiGraph

function test_allocations( g, edges, add, rem)
    for i::Int64 in edges, j::Int64 in add
        if  i == j
            continue
        end
        add_edge!(g, i, j)
    end

    for i::Int64 in edges, j::Int64 in rem 
        if  i == j
            continue
        end
        has_edge(g, i, j)
    end

    for i::Int64 in edges, j::Int64 in rem
        if  i == j
            continue
        end
        rem_edge!(g, i, j)
    end
end

println( "===================================================\n" );

g = Graphs.SimpleGraphs.SimpleDiGraph(1000) 

@time test_allocations(g, 1:1000, 1:30, 11:30)

g = Graphs.SimpleGraphs.SimpleDiGraph(1000) 
@time test_allocations( g, 1:1000, 1:30, 11:30)

println( "===================================================\n" );

#print( names( MaxDegreeDiGraph ) )

#print( methods( add_edge! ) )
degree = 50
g = MaxDegDiGraph(1000, degree)
@time test_allocations( g, 1:1000, 1:30, 11:30 )  # warm start


g = MaxDegDiGraph(1000, degree)
@time test_allocations(  g, 1:1000, 1:50, 11:30 )  # warm start
println( ne( g ) )

for  e ∈ edges( g )
    println( e )
end
