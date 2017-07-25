import os
def get_search_locations():
    # Find chronological ordering of the HDDs
    unordered_hdds = ['/media/michael/Elements/', '/media/michael/Elements1/', '/media/michael/Elements2/']
    ordered_hdds = [None, None, None]

    for hdd in unordered_hdds:
      if os.path.exists(hdd + 'Fieldwork_Data/2013/') and os.path.exists(hdd + 'Fieldwork_Data/2014/'):
          ordered_hdds[0] = hdd
      elif os.path.exists(hdd + 'Fieldwork_Data/2014/') and os.path.exists(hdd + 'Fieldwork_Data/2015/'):
          ordered_hdds[1] = hdd
      elif os.path.exists(hdd + 'Fieldwork_Data/2015') and not os.path.exists(hdd + 'Fieldwork_Data/2014'):
          ordered_hdds[2] = hdd

    print ordered_hdds


    # # Define all search locations
    search_locations = [
      (0, ordered_hdds[0] + 'Fieldwork_Data/2013/'),
      (0, ordered_hdds[0] + 'Fieldwork_Data/2014/'),
      (1, ordered_hdds[1] + 'Fieldwork_Data/2014/'),
      (1, ordered_hdds[1] + 'Fieldwork_Data/2015/'),
      (2, ordered_hdds[2] + 'Fieldwork_Data/2015/')
    ]

    return search_locations
