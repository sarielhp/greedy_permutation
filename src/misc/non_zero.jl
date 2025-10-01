"""
    NonZeroIterator{T, M<:AbstractMatrix{T}}

An iterator that yields only the non-zero entries of a given matrix `A`.

The iteration proceeds by checking the matrix elements in column-major order 
(linear indexing).
"""
struct NonZeroIterator{T, M<:AbstractMatrix{T}}
    A::M
    len::Int
end

# A simple constructor for convenience
NonZeroIterator(A::AbstractMatrix) = NonZeroIterator(A, length(A))

# 1. Define the iteration protocol using a single Base.iterate function
# The state is the linear index (starting at 1) of the matrix element
# we should check next.
function Base.iterate(iter::NonZeroIterator, state::Int=1)
    A = iter.A
    len = iter.len
    i = state

    # Loop from the current state index up to the total length of the matrix
    while i <= len
        val = A[i]
        
        # Use !iszero() for robust checking against the matrix's element type.
        if !iszero(val)
            # Found a non-zero element: return (value, next_state_index)
            # The next state is i + 1, which is the index immediately following the current element.
            return (val, i + 1)
        end
        
        # If the element is zero, move to the next index
        i += 1
    end
    
    # If the loop finishes without finding a non-zero element, the iteration is complete
    return nothing
end

# 2. Define Base.eltype and Base.length for the iterator (good practice)
Base.eltype(::Type{NonZeroIterator{T, M}}) where {T, M} = T
Base.length(iter::NonZeroIterator) = count(!iszero, iter.A)

# --- Example Usage ---

# 1. Create a sample matrix
A = [
    1.0 0.0 3.0;
    0.0 5.0 0.0;
    7.0 0.0 9.0
]

println("--- Original Matrix A ---")
display(A)

# 2. Create the iterator
nz_iter = NonZeroIterator(A)

# 3. Iterate and print the non-zero entries
println("\n--- Non-Zero Entries ---")
for val in nz_iter
    println(val)
end

# 4. We can also collect the results into an array
non_zero_array = collect(nz_iter)
println("\n--- Collected Array ---")
println(non_zero_array)

