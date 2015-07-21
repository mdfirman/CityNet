def meta_to_excel(meta, filename=None):
    """
    meta (dict(dict)):
        meta data as used in AzurePipeline

    filename (string):
        default: 'meta.xlsx'
    """
    if filename is None:
        filename = 'meta.xlsx'

    import pandas as pd
    df = pd.DataFrame(meta)
    df.to_excel('meta.xlsx')
    

def meta_to_excel(filename=None):
    """
    meta (dict(dict)):
        meta data as used in AzurePipeline

    filename (string):
        default: 'meta.xlsx'
    """
    if filename is None:
        filename = 'meta.xlsx'

    import pandas as pd
    return pd.read_excel('test.xlsx').to_dict()

