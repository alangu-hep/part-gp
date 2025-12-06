import uproot

def _write_trees(path, tables, treenames=['Events'], compression=-1, step=1048576):
    '''
    Adapted from Weaver
    '''
    if compression == -1:
        compression = uproot.LZ4(4)
    
    with uproot.recreate(file, compression=compression) as fout:
        
            tree = fout.mktree(treename, {k: table[k].type for k in table.fields})
            start = 0
            while start < len(table[table.fields[0]]) - 1:
                tree.extend({k: table[k][start:start + step] for k in table.fields})
                start += step