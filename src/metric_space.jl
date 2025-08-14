#! /bin/env julia

push!(LOAD_PATH, pwd()*"/src/cg/")
push!(LOAD_PATH, pwd()*"/src/" )

using FrechetDist;
using FrechetDist.cg;
using FrechetDist.cg.polygon;
using FrechetDist.cg.point;
using Graphs;
using Distances;


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


function  (@main)(args)

    if  ( length( args ) > 0 )
        m = read_fvecs( args[ 1 ] );
        d::Int64 = size( m, 1 );
        n::Int64 = size( m, 2 );
        println( "Dimension: ", d );
        println( "n        : ", n );
        
        PS = MPointsSpace( m );
        I, D = greedy_permutation_naive( PS )

        for i ∈ 1:n
            println( i, ":", I[i], "  D: ", D[i ] );
        end
        
        exit
    end

    
    n = 10;
    P = Polygon_random( 2, Float64, n );

    println( length( args ) )

    println( "Hello world!" );
    PS = PointsSpace( Points( P ) );
    I, D = greedy_permutation_naive( PS )

    for i ∈ 1:n
        println( i, ":", I[i], "  D: ", D[i ] );
    end

    return  0;
end

