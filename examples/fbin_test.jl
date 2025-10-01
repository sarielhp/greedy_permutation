#! /bin/env julia


#include(joinpath(@__DIR__, "..", "src", "read_fbin.jl"))

#push!(LOAD_PATH, pwd()*"/src/" )
include( "read_fbin.jl" )

function  (@main)(args)
    if  length( args ) == 0
        return 0
    end

    #println( "Reading file...", args[ 1 ] )
    @time m = read_fbin( args[ 1 ] )
    @time mm = mmap_read_fbin( args[ 1 ] )
    println( typeof( mm ) )
    println( "---------------------------------------" )
    println( typeof( m ) )
    println( size( m  ) )
    println( m[:, 1] )
    println( "---------------------------------------" )
    println( typeof( mm ) )
    println( size( mm  ) )
    println( mm[:, 1] )
    println( "---------------------------------------" )
    println( m == mm )
    
end
