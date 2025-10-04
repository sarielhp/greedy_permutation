############################################################################

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

function delete_fast!(vec::Vector, pos)
    vec[ pos ] = vec[ end ]
    
    pop!(vec)

    return vec
end

function find_first_available_file( env, out_dir )
    existing_files = readdir( out_dir )

#    println( existing_files )
#    exit( -1 );
    while  true
        env.out_file_count += 1;
        i = env.out_file_count += 1;

        # Use `@sprintf` to format the integer `i` as a 5-digit string,
        # padding with leading zeros if necessary.
        bfilename = @sprintf("%08d.txt", i )
        filename = @sprintf("%s/%s", out_dir, bfilename )

        #println( "exists? ", filename in existing_files )
        if !(filename in existing_files)
            return filename
        end
    end

    return nothing
end

function create_dir_from_file(filename)
    dir_path = dirname(filename)
    if !isempty(dir_path)
        mkpath(dir_path)
    end
end
