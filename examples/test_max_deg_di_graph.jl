#! /bin/env julia

using  BenchmarkTools
#using  BoundedDegreeGraphs
#using  Graphs
using  Graphs, Graphs.SimpleGraphs
#using Profile
#using PProf

# --- Example Usage (outside the module for clarity) ---

push!(LOAD_PATH, pwd()*"/src/" )

#using Graphs
using MaxDegreeDiGraph # Uncomment if this code was split into a file and a script
#using .MaxDegreeDiGraph # Uncomment if this code was split into a file and a script

function  old_example()
    # # Create a graph with 5 vertices and max out-degree 2 (d=2)
    g = MaxDegDiGraph(5, 2)
    
    println("Number of vertices: ", MaxDegreeDiGraph.nv(g))
    println("Initial number of edges: ", MaxDegreeDiGraph.ne(g)) # Expected: 0
    
    # # Add edges to vertex 1
    MaxDegreeDiGraph.add_edge!(g, 1, 2) # Success: 1 -> 2
    MaxDegreeDiGraph.add_edge!(g, 1, 3) # Success: 1 -> 3
    MaxDegreeDiGraph.add_edge!(g, 1, 4) # Failure: Degree limit reached (d=2)
    
    # # Add an edge to vertex 2
    MaxDegreeDiGraph.add_edge!(g, 2, 5) # Success: 2 -> 5
    
    println("\nAdjacency Matrix (0 = no edge):")
    println(g.adj_matrix)
    
    println("\nOut-neighbors of vertex 1: ", MaxDegreeDiGraph.outneighbors(g, 1)) # Expected: [2, 3]
    println("Out-neighbors of vertex 2: ", MaxDegreeDiGraph.outneighbors(g, 2)) # Expected: [5]
    println("Current number of edges: ", MaxDegreeDiGraph.ne(g)) # Expected: 3
    
    # # Remove an edge
    MaxDegreeDiGraph.rem_edge!(g, 1, 2) # Success
    MaxDegreeDiGraph.rem_edge!(g, 1, 5) # Failure (edge 1->5 does not exist)
    
    println("\nOut-neighbors of vertex 1 after removal: ", MaxDegreeDiGraph.outneighbors(g, 1)) # Expected: [3]
    println("Current number of edges: ", MaxDegreeDiGraph.ne(g)) # Expected: 2
    
end

# testing for allocations
function test_allocations( g, edges, add, rem)
    println( "\n\n\n", typeof( g ) );
    println( "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1111" )
    for i::Int64 in edges, j::Int64 in add
        #println( typeof( g ) )
        #println( typeof( i ) )
        #println( typeof( j ) )
        if  i == j
            continue
        end
        Graphs.SimpleGraphs.add_edge!(g, i, j)
    end
    println( "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1111" )

    for i::Int64 in edges, j::Int64 in rem 
        if  i == j
            continue
        end
        Graphs.SimpleGraphs.has_edge(g, i, j)
    end

    for i::Int64 in edges, j::Int64 in rem
        if  i == j
            continue
        end
        Graphs.SimpleGraphs.rem_edge!(g, i, j)
    end

end
function test_allocations_m( g, edges, add, rem)
    for i::Int64 in edges, j::Int64 in add
        if  i == j
            continue
        end
        MaxDegreeDiGraph.add_edge!(g, i, j)
    end
    println( "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1111" )

    for i::Int64 in edges, j::Int64 in rem 
        if  i == j
            continue
        end
        MaxDegreeDiGraph.has_edge(g, i, j)
    end

    for i::Int64 in edges, j::Int64 in rem
        if  i == j
            continue
        end
        MaxDegreeDiGraph.rem_edge!(g, i, j)
    end
end

println( "===================================================\n" );

g = Graphs.SimpleGraphs.SimpleDiGraph(1000) 
#println( typeof( g ) )
#add_edge!( g, SimpleEdge{Int}(2,3) )

@time test_allocations(g, 1:1000, 1:30, 11:30)

g = Graphs.SimpleGraphs.SimpleDiGraph(1000) 
@time test_allocations( g, 1:1000, 1:30, 11:30)

println( "===================================================\n" );

#print( names( MaxDegreeDiGraph ) )

#print( methods( add_edge! ) )
degree = 50
g = MaxDegDiGraph(1000, degree)
@time test_allocations_m( g, 1:1000, 1:30, 11:30 )  # warm start


g = MaxDegDiGraph(1000, degree)
@time test_allocations_m(  g, 1:1000, 1:50, 11:30 )  # warm start
