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
    df.to_excel(filename)
    

def excel_to_meta(filename=None):
    """
    meta (dict(dict)):
        meta data as used in AzurePipeline

    filename (string):
        default: 'meta.xlsx'
    """
    if filename is None:
        filename = 'meta.xlsx'

    import pandas as pd
    return pd.read_excel(filename).to_dict()

