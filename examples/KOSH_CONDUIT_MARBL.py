#!/usr/bin/env python
# coding: utf-8

# In[1]:


import kosh


# In[2]:


import sys
sys.path.append("/g/g19/cdoutrix/pipenvs/conduit/python-modules")


# In[3]:


import conduit
import conduit.relay
import conduit.blueprint


# In[4]:


conduit.about()


# In[5]:


from kosh import KoshLoader
import numpy

# Make sure local file is new sql file
kosh_example_sql_file = "my_store.sql"
    
# Create db on file
store = kosh.create_new_db(kosh_example_sql_file)

try:
    del store.loaders['sidre_mesh_blueprint_fields']
except:
    pass

import kosh.loaders.sidre
store.add_loader(kosh.loaders.sidre.SidreMeshBlueprintFieldLoader)


# In[6]:



sample = store.create(name="example", metadata={'project':"example"})
    
 #Associate file with datasets
sample.associate("/p/lustre1/cdoutrix/MARBL/tgvortex-1p/marbl_0000000.root",
                  mime_type="sidre_mesh_blueprint_fields")


# In[7]:


sample.list_features()


# In[8]:


sample.get("blast/energy")


# In[9]:


sample.get("blast/energy_001")


# In[10]:


sample.get("blast/energy_001", format="sidre/path")


# In[11]:


import kosh.transformers.sidre
print(sample.get("blast/energy_001", transformers=[kosh.transformers.sidre.SidreFeatureMetrics()]))


# In[12]:


s2 = store.create(name="example2", metadata={'project':"example"})
    
 #Associate file with datasets
s2.associate("/p/lustre1/cdoutrix/MARBL/tgvortex-64p/marbl_0000000.root",
                  mime_type="sidre_mesh_blueprint_fields")


# In[13]:


s2.list_features()


# In[14]:


s2.get("miranda/u")

