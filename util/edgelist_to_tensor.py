import numpy as np
import pandas as pd

def edgelist_to_tensor( edgelist ):

    # Minimum columns required in edgelist
    req_cols = pd.Series(['from','to','time'])
    if not np.all( req_cols.isin(edgelist.columns) ):
        raise NameError(f'edgelist must have columns {req_cols}')

    ### Identify all nodes, times
    all_nodes = np.union1d( edgelist['from'].unique(), edgelist['to'].unique())
    all_times = np.sort( edgelist['time'].unique() )

    # We will take as link values anything additional to these colums
    all_layers = np.setdiff1d( edgelist.columns, list(req_cols)+['layer'] )
    if all_layers.shape[0]==0:
        raise NameError(f'edgelist must have columns {req_cols} and at least one more numeric columns')
    print(f'The following columns will be taken as link values:\n{all_layers}')

    edgelist_temp = pd.MultiIndex.from_product([all_nodes, all_nodes, all_times], names = ["from", "to", "time"] )
    edgelist_temp = pd.DataFrame( index = edgelist_temp).reset_index()
    net_np = edgelist_temp.merge( edgelist, how='left' )

    ### Create tensor/array with Dynamic Network data ###
    net_np = net_np.pivot_table( index = ['from','to','time'],
                                values = all_layers.tolist(),
                                aggfunc='sum', fill_value=0, dropna=False )
    net_np = net_np.values.reshape( all_nodes.size, all_nodes.size, all_times.size, all_layers.size ).astype(float)

    return (net_np, [all_nodes, all_times, all_layers] )
