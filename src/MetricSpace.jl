### Modules for dealing with abstract finite metric spaces

module  MetricSpace

using Distances

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
        return  0.0
    end
    return  Dist( P.P[ x ], P.P[ y ] )
end

function dist_real( P::PointsSpace{PType}, x, y_real ) where {PType}
    return  Dist( P.P[ x ], y_real )
end

function Base.size(P::PointsSpace{PType} ) where {PType}
    return  length( P.P )
end


####################################################################
# MPointsSpace: Metric space defined by a matrix.
# Columns are the points.
####################################################################
struct MPointsSpace{PType} <: AbstractFMetricSpace
    m::Matrix{PType}
    n::Int     # Number of points

    function  MPointsSpace( _m::Matrix, _n::Int = 0 )
        if  _n == 0
            _n = size( _m, 2 )
        end

        mps = new{eltype(_m)}( _m, _n )
        @assert( 0 < mps.n <= size( _m, 2 ) )
        
        return  mps
    end
end


#function  MPointsSpace{PType}( _m::Matrix{PType} ) where {PType}
#    return  MPointsSpace( _m, size( _m, 2 ) )
#end
    

function dist( P::MPointsSpace{PType}, x, y ) where {PType}
    if  ( x == y )
        return  0.0
    end
    return  euclidean( P.m[ :, x ], P.m[ :, y ] )
end

function dist_real( P::MPointsSpace{PType}, x, y_real ) where {PType}
    return  euclidean( P.m[ :, x ], y_real )
end

function Base.size(P::MPointsSpace{PType} ) where {PType}
    return  P.n #size( P.m, 2 )
end

####################################################################
# PermutMetric: A permutation of a finite metric space.
####################################################################
struct PermutMetric{MetricType} <: AbstractFMetricSpace
    n::Int  # Number of points
    m::MetricType
    I::Vector{Int}
end

function  PermutMetric(_m::MetricType ) where {MetricType}
    _n = size( _m )
    return PermutMetric( _n, _m, [i for i ∈ 1:_n] )
end

function  PermutMetric(_m::MetricType, _I::Vector{Int} ) where {MetricType}
    return PermutMetric{MetricType}( size( _m ), _m, _I )
end


function dist( P::PermutMetric{MetricType}, x::Int, y::Int ) where {MetricType}
    if  ( x == y )
        return  0.0
    end
    return  dist( P.m, P.I[ x ], P.I[ y ] )
end

function dist_real( P::PermutMetric{MetricType}, x, y_real ) where {MetricType}
    return  dist_real( P.m, P.I[ x ], y_real )
end

function Base.size(P::PermutMetric{MetricSpace} ) where {MetricSpace}
    return  P.n
end

function swap!( P::PermutMetric{MetricType}, x, y ) where {MetricType}
    if  x == y
        return
    end
    P.I[ x ], P.I[ y ] = P.I[ y ], P.I[ x ]
end

######################################################################

function  update_distances( M::AbstractFMetricSpace, I, D, pos, n )
    x = I[ pos ]
    for  i ∈ pos+1:n
        y = I[ i ]
        d = dist( M, x, y )
        #println( "d: ", d, "  x: ", x, "  y: ", y, "  pos: ", pos, "  i: ", i )
        if   ( ( d < D[ i ] )  ||  ( D[ i ] < zero(Float64) ) )
            D[ i ] = d
        end
    end
end


function  greedy_permutation_naive( M::AbstractFMetricSpace, n::Int )
    #n = size( M )
    println( "m: ", n )
    I = [i for i ∈ 1:size(M) ]   ## Imprtant: n might be smaller than size(M)
    D = fill( Float64(-1.0), n )

    # Initialize
    update_distances( M, I, D, 1, n )

    for  i ∈ 2:n
        #println( "I : ", i )
        pos = argmax( D[ i:n ] ) + i - 1
        D[ i - 1 ] = D[ pos ]
        I[ i ], I[ pos ] =  I[ pos ], I[ i ]
        #Centers[ i ], Centers[ pos ] =  Centers[ pos ], Centers[ i ]
        D[ i ], D[ pos ] =  D[ pos ], D[ i ]
        update_distances( M, I, D, i, n )
    end
    D[ n ] = zero( Float64 )
    
    return  I, D
end


export AbstractFMetricSpace, PointsSpace, greedy_permutation_naive
export MPointsSpace

export dist, dist_real 

# PermutMetric: Allows permuting (renaming) vertices in the metric on
# the fly. swap! performs the renaming.
export  PermutMetric 
export  swap!

end
