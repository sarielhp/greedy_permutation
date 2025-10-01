#!


function  test_nn_queries_2( fn_data, fn_queries )
    m_i = read_fvecs( fn_data );
    m_q = read_fvecs( fn_queries );
    d::Int64 = size( m_i, 1 );
    n::Int64 = size( m_i, 2 );
    n_q::Int64 = size( m_q, 2 );
    println( "Dimension: ", d );
    println( "n        : ", n );
    println( "n_q      : ", n_q );
        
    
    PS = MPointsSpace( m_i );    
    mp_rand = PermutMetric( PS );    
    #G_rand = NNGraph( mp_rand, n );

    #println( "Computing random DAG" );
    #nng_random_dag!( G_rand, n, 10 );

    QS = MPointsSpace( m_q );
    for i ∈ 1:n_q
        println( "Query  ", i, ": " );
        println( "rand       : " );
        
        if  ( nng_nn_search_print( G_rand, QS.m[ :, i ], 20 ) )
            break;
        end
        nng_nn_search_print( G_rand, QS.m[ :, i ], 40 );
        nng_nn_search_print( G_rand, QS.m[ :, i ], 80 );
        nng_nn_search_print( G_rand, QS.m[ :, i ], 400 );
        nng_nn_search_print( G_rand, QS.m[ :, i ], 800 );
    end

end
