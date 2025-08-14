#! /bin/env julia

mutable struct  ListsInt
   n::Int
   L::Vector{Int}
end

function  ListInt( _n::Int )
    return  ListsInt( _n, zeros( Int, n ) )
end


