#! /bin/env julia

using FileIO
using LinearAlgebra
using Statistics
using Mmap

### Seems unstable under WSL. Yuk.
function mmap_read_fbin( filename::String )
    # Gets dimensions...
    f = open(filename, "r")
    nvecs = read(f, Int32)
    dim = read(f, Int32)
    close( f )

    # Now, read the matrix using memory mapping.
    # The 'offset' argument in Mmap.mmap is the starting byte offset.
    # We need to specify the element type (Float64) and the dimensions (3, 4).
    m = Mmap.mmap( filename, Matrix{Float32}, (dim, nvecs),
                                 8 )
    return  m
end

function read_fbin(filename::String,
                   start_idx::Int=0, chunk_size::Union{Int, Nothing}=nothing)
    """ Read *.fbin file that contains Float32 vectors
    Args:
        :param filename (String): path to *.fbin file
        :param start_idx (Int): start reading vectors from this index
        :param chunk_size (Int): number of vectors to read. 
                                 If nothing, read all vectors
    Returns:
        Array of Float32 vectors (Matrix{Float32})
    """
    f = open(filename, "r")
    nvecs = read(f, Int32)
    dim = read(f, Int32)
                                     
    seek(f, 8 + start_idx * sizeof(Float32) * dim)

    nvecs_to_read = isnothing(chunk_size) ? nvecs - start_idx : chunk_size
    
    arr = Vector{Float32}(undef, nvecs_to_read * dim)
    readbytes!(f, reinterpret(UInt8, arr))
    
    close(f)

    #println( "DIM: ", dim )

    #println( arr[ 1:3 ] )
    
    m = reshape(arr, ( dim, nvecs_to_read ) )

    #println( m[1,1] )
    #println( m[2,1] )
    println( "Rows : ", size( m, 1 ) )
    println( "Cols : ", size( m, 2 ) )

    return  m
end

function read_fbin_n(filename::String, n::Int = 0)
    f = open(filename, "r")
    nvecs = read(f, Int32)
    dim = read(f, Int32)

    if  n != 0 
        nvecs = min( nvecs, n )
    end
    seek(f, 8 )
    
    arr = Vector{Float32}(undef, nvecs * dim)
    readbytes!(f, reinterpret(UInt8, arr))
    
    close(f)
    m = reshape(arr, ( dim, nvecs ) )

    #println( "Rows : ", size( m, 1 ) )
    #println( "Cols : ", size( m, 2 ) )

    return  m
end


function read_ibin(filename::String, start_idx::Int=0, chunk_size::Union{Int, Nothing}=nothing)
    """ Read *.ibin file that contains Int32 vectors
    Args:
        :param filename (String): path to *.ibin file
        :param start_idx (Int): start reading vectors from this index
        :param chunk_size (Int): number of vectors to read.
                                 If nothing, read all vectors
    Returns:
        Array of Int32 vectors (Matrix{Int32})
    """
    f = open(filename, "r")
    nvecs = read(f, Int32)
    dim = read(f, Int32)

    seek(f, 8 + start_idx * sizeof(Int32) * dim)

    nvecs_to_read = isnothing(chunk_size) ? nvecs - start_idx : chunk_size
    
    arr = Vector{Int32}(undef, nvecs_to_read * dim)
    readbytes!(f, reinterpret(UInt8, arr))
    
    close(f)
    return reshape(arr, (dim, nvecs_to_read))'
end

function write_fbin(filename::String, vecs::Matrix{Float32})
    """ Write a matrix of Float32 vectors to *.fbin file
    Args:
        :param filename (String): path to *.fbin file
        :param vecs (Matrix{Float32}): matrix of Float32 vectors to write
    """
    f = open(filename, "w")
    nvecs, dim = size(vecs)
    
    write(f, Int32(nvecs))
    write(f, Int32(dim))
    write(f, vecs')
    
    close(f)
end


function write_ibin(filename::String, vecs::Matrix{Int32})
    """ Write a matrix of Int32 vectors to *.ibin file
    Args:
        :param filename (String): path to *.ibin file
        :param vecs (Matrix{Int32}): matrix of Int32 vectors to write
    """
    f = open(filename, "w")
    nvecs, dim = size(vecs)
    
    write(f, Int32(nvecs))
    write(f, Int32(dim))
    write(f, vecs')

    close(f)
end
