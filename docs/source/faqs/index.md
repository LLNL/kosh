# Frequently Asked Questions

## Can I make Kosh faster?

Yes. By default Kosh runs in "safe" mode, synchronizing every changes to the store

You can turn this off and synchronize only when you're done working:

```python
import kosh
store = kosh.KoshStore(db_uri="some_kosh_store.sql", sync=False)
# Some Kosh work here
store.sync()
```

If you opened the store in sync mode you can switch back and forth:

```python
import kosh
store = kosh.KoshStore(db_uri="some_kosh_store.sql")

# some synchronous work
store.synchronous(False)
# Some async work (faster)
# now let's go back to synchrouns mode
store.synchronous(True)

# You can switch back and forth w/o passing the mode
mode = store.synchronous()
# Let's query
print("Synchrononus mode?", mode, store.is_synchronous())
```

If you are associating multiple files with a dataset pass them all at once

Rather than
```python
for i in range(200):
    ds.associate(str(i), metadata={"name":str(i)}, mime_type="type_{}".format(i))
```

Consider
```python
ds.associate([str(i) for i in range(200)], metadata = [ {"name":str(i)} for i in range(200)], mime_type=["type_{}".format(i) for i in range(200) ])
```

