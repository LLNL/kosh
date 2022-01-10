import os
import uuid
import time
import warnings
import sina
from sina.model import Record
from .core_sina import KoshSinaObject
from .utils import get_graph
from .utils import compute_fast_sha
from .utils import compute_long_sha
from .utils import cleanup_sina_record_from_kosh_sync
from .utils import update_json_file_with_records_and_relationships
import kosh
try:
    import orjson
except ImportError:
    import json as orjson  # noqa
try:
    basestring
except NameError:
    basestring = str


class KoshDataset(KoshSinaObject):
    def __init__(self, id, store, schema=None, record=None, kosh_type=None):
        """KoshSinaDataset Sina representation of Kosh Dataset

        :param id: dataset's unique Id
        :type id: str
        :param store: store containing the dataset
        :type store: KoshSinaStore
        :param schema: Kosh schema validator
        :type schema: KoshSchema
        :param record: to avoid looking up in sina pass sina record
        :type record: Record
        :param kosh_type: type of Kosh object (dataset, file, project, ...)
        :type kosh_type: str
        """
        if kosh_type is None:
            kosh_type = store._dataset_record_type
        super(KoshDataset, self).__init__(id, kosh_type=kosh_type,
                                          protected=[
                                              "__name__", "__creator__", "__store__",
                                              "_associated_data_", "__features__"],
                                          record_handler=store.__record_handler__,
                                          store=store, schema=schema, record=record)
        self.__dict__["__record_handler__"] = store.__record_handler__
        self.__dict__["__features__"] = {None: {}}
        if record is None:
            record = self.get_record()
        try:
            self.__dict__["__creator__"] = record["data"]["creator"]["value"]
        except Exception:
            pass
        try:
            self.__dict__["__name__"] = record["data"]["name"]["value"]
        except Exception:
            pass
        if schema is not None or "schema" in record["data"]:
            self.validate()

    def __str__(self):
        """string representation"""
        st = ""
        st += "KOSH DATASET\n"
        st += "\tid: {}\n".format(self.id)
        try:
            st += "\tname: {}\n".format(self.__name__)
        except Exception:
            st += "\tname: ???\n"
        try:
            st += "\tcreator: {}\n".format(self.creator)
        except Exception:
            st += "\tcreator: ???\n"
        atts = self.__attributes__
        if len(atts) > 0:
            st += "\n--- Attributes ---\n"
            for a in sorted(atts):
                if a == "_associated_data_":
                    continue
                st += "\t{}: {}\n".format(a, atts[a])
        if self._associated_data_ is not None:
            st += "--- Associated Data ({})---\n".format(
                len(self._associated_data_))
            # Let's organize per mime_type
            associated = {}
            for a in self._associated_data_:
                if a == self.id:
                    st2 = "internal ( {} )".format(
                        ", ".join(self.get_record()["curve_sets"].keys()))
                    if "sina/curve" not in associated:
                        associated["sina/curve"] = [st2, ]
                    else:
                        associated["sina/curve"].append(st2)
                else:
                    if "__uri__" in a:
                        a_id, a_uri = a.split("__uri__")
                        a_mime_type = self.get_record(
                        )["files"][a_uri]["mimetype"]
                    else:
                        a_id, a_uri = a, None
                    a_obj = self.__store__._load(a_id)
                    if a_uri is None:
                        a_uri = a_obj.uri
                        a_mime_type = a_obj.mime_type
                    st2 = "{a_uri} ( {a} )".format(a_uri=a_uri, a=a_id)
                    if a_mime_type not in associated:
                        associated[a_mime_type] = [st2, ]
                    else:
                        associated[a_mime_type].append(st2)
            for mime in sorted(associated):
                st += "\tMime_type: {mime}".format(mime=mime)
                for uri in sorted(associated[mime]):
                    st += "\n\t\t{uri}".format(uri=uri)
                st += "\n"
        ensembles = list(self.get_ensembles(ids_only=True))
        st += "--- Ensembles ({})---".format(len(ensembles))
        st += "\n\t"+str([str(x) for x in ensembles])
        return st

    def _repr_pretty_(self, p, cycle):
        """Pretty display in Ipython"""
        p.text(self.__str__())

    def cleanup_files(self, dry_run=False, interactive=False, clean_fastsha=False, **search_keys):
        """Cleanup the dataset from references to dead files
        Also updates the fast_shas if necessary
        You can filter associated objects by passing key=values
        e.g mime_type=hdf5 will only dissociate non-existing files associated with mime_type hdf5
        some_att=some_val will only dissociate non-existing files associated and having the attribute
        'some_att' with value of 'some_val'
        returns list of uris to be removed.
        :param dry_run: Only does a dry_run
        :type dry_run: bool
        :param interactive: interactive mode, ask before dissociating
        :type interactive: bool
        :param clean_fastsha: Do we want to update fast_sha if it changed?
        :type clean_fastsha: bool
        :returns: list of uris (to be) removed or updated
        :rtype: list
        """
        bads = []
        for associated in self.find(**search_keys):
            clean = 'n'
            if not os.path.exists(associated.uri):  # Ok this is gone
                bads.append(associated.uri)
                if dry_run:  # Dry run
                    clean = 'n'
                elif interactive:
                    clean = input("\tDo you want to dissociate {} (mime_type: {})? [Y/n]".format(
                        associated.uri, associated.mime_type)).strip()
                    if len(clean) > 0:
                        clean = clean[0]
                        clean = clean.lower()
                    else:
                        clean = 'y'
                else:
                    clean = 'y'
                if clean == 'y':
                    self.dissociate(associated.uri)
            elif clean_fastsha:
                # file still exists
                # We might want to update its fast sha
                fast_sha = compute_fast_sha(associated.uri)
                if fast_sha != associated.fast_sha:
                    bads.append(associated.uri)
                    if dry_run:  # Dry run
                        clean = 'n'
                    elif interactive:
                        clean = input("\tfast_sha for {} seems to have changed from {}"
                                      " to {}, do you wish to update?".format(
                                          associated.uri, associated.fast_sha, fast_sha))
                        if len(clean) > 0:
                            clean = clean[0]
                            clean = clean.lower()
                        else:
                            clean = 'y'
                    else:
                        clean = "y"
                    if clean == "y":
                        associated.fast_sha = fast_sha
        return bads

    def check_integrity(self):
        """Runs a sanity check on the dataset:
        1- Are associated files reachable?
        2- Did fast_shas change since file was associated
        """
        return self.cleanup_files(dry_run=True, clean_fastsha=True)

    def open(self, Id=None, loader=None, *args, **kargs):
        """open an object associated with a dataset

        :param Id: id of object to open, defaults to None which means first one.
        :type Id: str, optional
        :param loader: loader to use for this object, defaults to None
        :type loader: KoshLoader, optional
        :raises RuntimeError: object id not associated with dataset
        :return: object ready to be used
        """
        if Id is None:
            if len(self._associated_data_) > 0:
                Id = self._associated_data_[0]
            else:
                for Id in self._associated_data_:
                    return self.__store__.open(Id, loader)
        elif Id not in self._associated_data_:
            raise RuntimeError(
                "object {Id} is not associated with this dataset".format(
                    Id=Id))
        return self.__store__.open(Id, loader, *args, **kargs)

    def list_features(self, Id=None, loader=None,
                      use_cache=True, *args, **kargs):
        """list_features list features available if multiple associated data lead to duplicate feature name
        then the associated_data uri gets appended to feature name

        :param Id: id of associated object to get list of features from, defaults to None which means all
        :type Id: str, optional
        :param loader: loader to use to search for feature, will return ONLY features that the loader knows about
        :type loader: kosh.loaders.KoshLoader
        :param use_cache: If features is found on cache use it (default: True)
        :type use_cache: bool
        :raises RuntimeError: object id not associated with dataset
        :return: list of features available
        :rtype: list
        """
        if use_cache and self.__dict__["__features__"].get(
                Id, {}).get(loader, None) is not None:
            return self.__dict__["__features__"][Id][loader]
        # Ok no need to sync any of this we will not touch the code
        saved_sync = self.__store__.is_synchronous()
        if saved_sync:
            # we will not update any rec in here, turning off sync
            # it makes things much faster
            backup = self.__store__.__sync__dict__
            self.__store__.__sync__dict__ = {}
            self.__store__.synchronous()
        features = []
        loaders = []
        associated_data = self._associated_data_
        if Id is None:
            for associated in associated_data:
                if loader is None:
                    ld, _ = self.__store__._find_loader(associated)
                else:
                    if associated not in self.__store__._cached_loaders:
                        self.__store__._cached_loaders[associated] = loader(
                            self.__store__._load(associated)), None
                    ld, _ = self.__store__._cached_loaders[associated]
                loaders.append(ld)
                try:
                    features += ld._list_features(*
                                                  args, use_cache=use_cache, **kargs)
                except Exception:  # Ok the loader couldn't get the feature list
                    pass
            if len(features) != len(set(features)):
                # duplicate features we need to redo
                # Adding uri to feature name
                ided_features = []
                for index, associated in enumerate(associated_data):
                    ld = loaders[index]
                    if ld is None:
                        continue
                    sp = associated.split("__uri__")
                    if len(sp) > 1:
                        uri = sp[1]
                    else:
                        uri = ld.uri
                    these_features = ld._list_features(
                        *args, use_cache=use_cache, **kargs)
                    for feature in these_features:
                        if features.count(feature) > 1:  # duplicate
                            ided_features.append(
                                "{feature}_@_{uri}".format(feature=feature, uri=uri))
                        else:  # not duplicate name
                            ided_features.append(feature)
                features = ided_features
        elif Id not in self._associated_data_:
            raise RuntimeError(
                "object {Id} is not associated with this dataset".format(
                    Id=Id))
        else:
            ld, _ = self.__store__._find_loader(Id)
            features = ld._list_features(*args, use_cache=use_cache, **kargs)
        features_id = self.__dict__["__features__"].get(Id, {})
        features_id[loader] = features
        self.__dict__["__features__"][Id] = features_id
        if saved_sync:
            # we need to restore sync mode
            self.__store__.__sync__dict__ = backup
            self.__store__.synchronous()
        return features

    def get_execution_graph(self, feature=None, Id=None,
                            loader=None, transformers=[], *args, **kargs):
        """get data for a specific feature
        :param feature: feature (variable) to read, defaults to None
        :type feature: str, optional if loader does not require this
        :param Id: object to read in, defaults to None
        :type Id: str, optional
        :param loader: loader to use to get data,
                       defaults to None means pick for me
        :type loader: kosh.loaders.KoshLoader
        :param transformers: A list of transformers to use after the data is loaded
        :type transformers: kosh.transformer.KoshTranformer
        :returns: [description]
        :rtype: [type]
        """
        if feature is None:
            out = []
            for feat in self.list_features():
                out.append(self.get_execution_graph(Id=None,
                                                    feature=feat,
                                                    format=format,
                                                    loader=loader,
                                                    transformers=transformers,
                                                    *args, **kargs))
            return out
        # Need to make sure transformers are a list
        if not isinstance(transformers, (list, tuple)):
            transformers = [transformers, ]
        # we need to figure which associated data has the feature
        if not isinstance(feature, list):
            features = [feature, ]
        else:
            features = feature
        possibles = {}
        inter = None
        union = set()
        for feature_ in features:
            possible_ids = []
            if Id is None:
                for a in self._associated_data_:
                    a_original = a
                    if "__uri__" in a:
                        # Ok this is a pure sina file with mime_type
                        a, _ = a.split("__uri__")
                    a_obj = self.__store__._load(a)
                    if loader is None:
                        ld, _ = self.__store__._find_loader(a_original)
                        if ld is None:  # unknown mimetype probably
                            continue
                    else:
                        if a_obj.mime_type in loader.types:
                            ld = loader(a_obj)
                        else:
                            continue
                    # Dataset with curve have themselves as uri
                    obj_uri = getattr(ld.obj, "uri", "self")
                    if ("_@_" not in feature_ and feature_ in ld._list_features()) or\
                            feature_ is None or\
                            (feature_[:-len(obj_uri) - 3] in ld._list_features() and
                             feature_[-len(obj_uri):] == obj_uri):
                        possible_ids.append(a_original)
                if possible_ids == []:  # All failed but could be something about the feature
                    raise ValueError(
                        "Cannot find feature {} in dataset".format(feature_))
            elif Id == self.id:
                # Ok asking for data not associated externally
                # Likely curve
                ld, _ = self.__store__._find_loader(Id)
                if feature_ in ld._list_features():
                    possible_ids = [Id, ]
                else:  # ok not a curve maybe a file?
                    rec = self.get_record()
                    for uri in rec["files"]:
                        if "mimetype" in rec["files"][uri]:
                            full_id = "{}__uri__{}".format(Id, uri)
                            ld, _ = self.__store__._find_loader(full_id)
                            if ld is not None and feature_ in ld.list_features():
                                possible_ids = [full_id, ]
            elif Id not in self._associated_data_:
                raise RuntimeError(
                    "object {Id} is not associated with this dataset".format(
                        Id=Id))
            else:
                possible_ids = [Id, ]
            if inter is None:
                inter = set(possible_ids)
            else:
                inter = inter.intersection(set(possible_ids))
            union = union.union(set(possible_ids))
            possibles[feature_] = possible_ids

        if len(inter) != 0:
            union = inter

        ids = {}
        # Now let's go through each possible uri
        # and group features in them
        for id_ in union:
            matching_features = []
            for feature_ in features:
                if feature_ in possibles and id_ in possibles[feature_]:
                    matching_features.append(feature_)
                    del(possibles[feature_])
            if len(matching_features) > 0:
                ids[id_] = matching_features

        out = []
        for id_ in ids:
            features = ids[id_]
            for Id in possible_ids:
                tmp = None
                try:
                    if loader is None:
                        ld, _ = self.__store__._find_loader(Id)
                        mime_type = ld._mime_type
                    else:
                        if Id not in self.__store__._cached_loaders:
                            a_obj = self.__store__._load(Id)
                            self.__store__._cached_loaders[Id] = loader(a_obj)
                            mime_type = a_obj.mime_type
                        ld = self.__store__._cached_loaders[Id]
                    # Essentially make a copy
                    # Because we want to attach the feature to it
                    # But let's not lose the cached list_features
                    saved_listed_features = ld.__dict__[
                        "_KoshLoader__listed_features"]
                    ld_uri = getattr(ld, "uri", None)
                    ld = ld.__class__(
                        ld.obj, mime_type=ld._mime_type, uri=ld_uri)
                    ld.__dict__[
                        "_KoshLoader__listed_features"] = saved_listed_features
                    # Ensures there is a possible path to format
                    get_graph(mime_type, ld, transformers)
                    final_features = []
                    obj_uri = getattr(ld.obj, "uri", "")
                    for feature_ in features:
                        if (feature_[:-len(obj_uri) - 3] in ld._list_features()
                                and feature_[-len(obj_uri):] == obj_uri):
                            final_features.append(
                                feature_[:-len(obj_uri) - 3])
                        else:
                            final_features.append(feature_)
                    if len(final_features) == 1:
                        final_features = final_features[0]
                    tmp = ld.get_execution_graph(final_features,
                                                 transformers=transformers
                                                 )
                    ld.feature = final_features
                    ExecGraph = kosh.exec_graphs.KoshExecutionGraph(tmp)
                except Exception:
                    import traceback
                    traceback.print_exc()
                    ExecGraph = kosh.exec_graphs.KoshExecutionGraph(tmp)
                out.append(ExecGraph)

        if len(out) == 1:
            return out[0]
        else:
            return out

    def get(self, feature=None, format=None, Id=None, loader=None,
            group=False, transformers=[], *args, **kargs):
        """get data for a specific feature
        :param feature: feature (variable) to read, defaults to None
        :type feature: str, optional if loader does not require this
        :param format: desired format after extraction
        :type format: str
        :param Id: object to read in, defaults to None
        :type Id: str, optional
        :param loader: loader to use to get data,
                       defaults to None means pick for me
        :type loader: kosh.loaders.KoshLoader
        :param group: group multiple features in one get call, assumes loader can handle this
        :type group: bool
        :param transformers: A list of transformers to use after the data is loaded
        :type transformers: kosh.transformer.KoshTranformer
        :raises RuntimeException: could not get feature
        :raises RuntimeError: object id not associated with dataset
        :returns: [description]
        :rtype: [type]
        """
        G = self.get_execution_graph(
            feature=feature,
            Id=Id,
            loader=loader,
            transformers=transformers,
            *args,
            **kargs)
        if isinstance(G, list):
            return [g.traverse(format=format, *args, **kargs) for g in G]
        else:
            return G.traverse(format=format, *args, **kargs)

    def __getitem__(self, feature):
        """Shortcut to access a feature or list of
        :param feature: feature(s) to access in dataset
        :type feature: str or list of str
        :returns: (list of) access point to feature requested
        :rtype: (list of) kosh.execution_graph.KoshIoGraph
        """
        return self.get_execution_graph(feature)

    def __dir__(self):
        """__dir__ list functions and attributes associated with dataset
        :return: functions, methods, attribute associated with this dataset
        :rtype: list
        """
        current = set(super(KoshDataset, self).__dir__())
        try:
            atts = set(self.listattributes() + self.__protected__)
        except Exception:
            atts = set()
        return list(current.union(atts))

    def reassociate(self, target, source=None, absolute_path=True):
        """This function allows to re-associate data whose uri might have changed

        The source can be the original uri or sha and target is the new uri to use.
        :param target: New uri
        :type target: str
        :param source: uri or sha (long or short of reassociate)
                       to reassociate with target, if None then the short uri from target will be used
        :type source: str or None
        :param absolute_path: if file exists should we store its absolute_path
        :type absolute_path: bool
        :return: None
        :rtype: None
        """
        # First let's convert to abs path if necessary
        if absolute_path:
            if os.path.exists(target):
                target = os.path.abspath(target)
            if source is not None and os.path.exists(source):
                source = os.path.abspath(source)

        # Now, did we pass a source for uri to replace?
        if source is None:
            source = compute_fast_sha(target)

        # Ok now let's get all associated uri that match
        # Fist assuming it's a fast_sha
        matches = list(self.find(fast_sha=source, ids_only=True))
        print("MATCHES FROM SHA:", matches)
        # Now it could be simply a uri
        matches += list(self.find(uri=source, ids_only=True))

        # And it's quite possible it's a long_sha too
        matches += list(self.find(long_sha=source, ids_only=True))

        # And now let's do the work
        for match_id in matches:
            match = self.__store__._load(match_id)
            match.uri = target

    def validate(self):
        """If dataset has a schema then make sure all attributes pass the schema"""
        if self.schema is not None:
            self.schema.validate(self)

    def searchable_source_attributes(self):
        """Returns all the attributes of associated sources
        :return: List of all attributes you can use to search sources in the dataset
        :rtype: set
        """
        searchable = set()
        for source in self.find():
            searchable = searchable.union(source.listattributes())
        return searchable

    def describe_feature(self, feature, Id=None, **kargs):
        """describe a feature

        :param feature: feature (variable) to read, defaults to None
        :type feature: str, optional if loader does not require this
        :param Id: id of associated object to get list of features from, defaults to None which means all
        :type Id: str, optional
        :param kargs: keywords to pass to list_features (optional)
        :type kargs: keyword=value
        :raises RuntimeError: object id not associated with dataset
        :return: dictionary describing the feature
        :rtype: dict
        """
        loader = None
        if Id is None:
            for a in self._associated_data_:
                ld, _ = self.__store__._find_loader(a)
                if feature in ld._list_features(**kargs) or \
                        (feature[:-len(ld.obj.uri) - 3] in ld._list_features()
                         and feature[-len(ld.obj.uri):] == ld.obj.uri):
                    loader = ld
                    break
        elif Id not in self._associated_data_:
            raise RuntimeError(
                "object {Id} is not associated with this dataset".format(
                    Id=Id))
        else:
            loader, _ = self.__store__._find_loader(Id)
        return loader.describe_feature(feature)

    def dissociate(self, uri, absolute_path=True):
        """dissociates a uri/mime_type with this dataset

        :param uri: uri to access file
        :type uri: str
        :param absolute_path: if file exists should we store its absolute_path
        :type absolute_path: bool
        :return: None
        :rtype: None
        """

        if absolute_path and os.path.exists(uri):
            uri = os.path.abspath(uri)
        rec = self.get_record()
        if uri not in rec["files"]:
            # Not associated with this uri anyway
            return
        kosh_id = str(rec["files"][uri]["kosh_id"])
        del(rec["files"][uri])
        now = time.time()
        rec["user_defined"]["{uri}___associated_last_modified".format(
            uri=uri)] = now
        if self.__store__.__sync__:
            self._update_record(rec)
        # Get all object that have been associated with this uri
        rec = self.__store__.get_record(kosh_id)
        if (not hasattr(rec, "associated")) or len(
                rec.associated) == 0:  # ok no other object is associated
            self.__store__.delete(kosh_id)
            if kosh_id in self.__store__._cached_loaders:
                del(self.__store__._cached_loaders[kosh_id])

        # Since we changed the associated, we need to cleanup
        # the features cache
        self.__dict__["__features__"][None] = {}
        self.__dict__["__features__"][kosh_id] = {}

    def associate(self, uri, mime_type, metadata={},
                  id_only=True, long_sha=False, absolute_path=True):
        """associates a uri/mime_type with this dataset

        :param uri: uri(s) to access content
        :type uri: str or list of str
        :param mime_type: mime type associated with this file
        :type mime_type: str or list of str
        :param metadata: metadata to associate with file, defaults to {}
        :type metadata: dict, optional
        :param id_only: do not return kosh file object, just its id
        :type id_only: bool
        :param long_sha: Do we compute the long sha on this or not?
        :type long_sha: bool
        :param absolute_path: if file exists should we store its absolute_path
        :type absolute_path: bool
        :return: A (list) Kosh Sina File(s)
        :rtype: list of KoshSinaFile or KoshSinaFile
        """

        rec = self.get_record()
        # Need to remember we touched associated files
        now = time.time()

        if isinstance(uri, basestring):
            uris = [uri, ]
            metadatas = [metadata, ]
            mime_types = [mime_type, ]
            single_element = True
        else:
            uris = uri
            if isinstance(metadata, dict):
                metadatas = [metadata, ] * len(uris)
            else:
                metadatas = metadata
            if isinstance(mime_type, basestring):
                mime_types = [mime_type, ] * len(uris)
            else:
                mime_types = mime_type
            single_element = False

        new_recs = []
        kosh_file_ids = []

        for i, uri in enumerate(uris):
            try:
                meta = metadatas[i].copy()
                if os.path.exists(uri):
                    if long_sha:
                        meta["long_sha"] = compute_long_sha(uri)
                    if absolute_path:
                        uri = os.path.abspath(uri)
                    if not os.path.isdir(uri) and "fast_sha" not in meta:
                        meta["fast_sha"] = compute_fast_sha(uri)
                rec["user_defined"]["{uri}___associated_last_modified".format(
                    uri=uri)] = now
                # We need to check if the uri was already associated somewhere
                tmp_uris = list(self.__store__.find(
                    types=[self.__store__._sources_type, ], uri=uri, ids_only=True))

                if len(tmp_uris) == 0:
                    Id = uuid.uuid4().hex
                    rec_obj = Record(id=Id, type=self.__store__._sources_type)
                else:
                    rec_obj = self.__store__.get_record(tmp_uris[0])
                    Id = rec_obj.id
                    existing_mime = rec_obj["data"]["mime_type"]["value"]
                    mime_type = mime_types[i]
                    if existing_mime != mime_types[i]:
                        rec["files"][uri]["mime_type"] = existing_mime
                        raise TypeError("source {} is already associated with another dataset with mimetype"
                                        " '{}' you specified mime_type '{}'".format(uri, existing_mime, mime_types[i]))
                rec.add_file(uri, mime_types[i])
                rec["files"][uri]["kosh_id"] = Id
                meta["uri"] = uri
                meta["mime_type"] = mime_types[i]
                meta["associated"] = [self.id, ]
                for key in meta:
                    rec_obj.add_data(key, meta[key])
                    last_modif_att = "{name}_last_modified".format(name=key)
                    rec_obj["user_defined"][last_modif_att] = time.time()
                if not self.__store__.__sync__:
                    rec_obj["user_defined"]["last_update_from_db"] = time.time()
                    self.__store__.__sync__dict__[Id] = rec_obj
                new_recs.append(rec_obj)
            except TypeError as err:
                raise(err)
            except Exception:
                # file already in there
                # Let's get the matching id
                if rec_obj["data"]["mime_type"]["value"] != mime_types[i]:
                    raise TypeError("file {} is already associated with this dataset with mimetype"
                                    " '{}' you specified mime_type '{}'".format(uri, existing_mime, mime_type))
                else:
                    Id = rec["files"][uri]["kosh_id"]
                    if len(metadatas[i]) != 0:
                        warnings.warn(
                            "uri {} was already associated, metadata will "
                            "stay unchanged\nEdit object (id={}) directly to update attributes.".format(uri, Id))
            kosh_file_ids.append(Id)

        if self.__store__.__sync__:
            self.__store__.lock()
            self.__store__.__record_handler__.insert(new_recs)
            self.__store__.unlock()
            self._update_record(rec)
        else:
            self._update_record(rec, self.__store__._added_unsync_mem_store)

        # Since we changed the associated, we need to cleanup
        # the features cache
        self.__dict__["__features__"][None] = {}

        if id_only:
            if single_element:
                return kosh_file_ids[0]
            else:
                return kosh_file_ids

        kosh_files = []
        for Id in kosh_file_ids:
            self.__dict__["__features__"][Id] = {}
            kosh_file = KoshSinaObject(Id=Id,
                                       kosh_type=self.__store__._sources_type,
                                       store=self.__store__,
                                       metadata=metadata,
                                       record_handler=self.__record_handler__)
            kosh_files.append(kosh_file)

        if single_element:
            return kosh_files[0]
        else:
            return kosh_files

    def search(self, *atts, **keys):
        """
        Deprecated use find
        """
        warnings.warn("The 'search' function is deprecated and now called `find`.\n"
                      "Please update your code to use `find` as `search` might disappear in the future",
                      DeprecationWarning)
        return self.find(*atts, **keys)

    def find(self, *atts, **keys):
        """find associated data matching some metadata
        arguments are the metadata name we are looking for e.g
        find("attr1", "attr2")
        you can further restrict by specifying exact value for a metadata
        via key=value
        you can return ids only by using: ids_only=True
        range can be specified via: sina.utils.DataRange(min, max)

        "file_uri" is a special key that will return the kosh object associated
        with this dataset for the given uri.  e.g store.find(file_uri=uri)

        :return: list of matching objects associated with dataset
        :rtype: list
        """

        if self._associated_data_ is None:
            return
        sina_kargs = {}
        ids_only = keys.pop("ids_only", False)
        # We are only interested in ids from Sina
        sina_kargs["ids_only"] = True

        inter_recs = self._associated_data_
        tag = "{}__uri__".format(self.id)
        tag_len = len(tag)
        virtuals = [x[tag_len:] for x in inter_recs if x.startswith(tag)]
        # Bug in sina 1.10.0 forces us to remove the virtual from the pool
        for v_id in virtuals:
            inter_recs.remove("{}{}".format(tag, v_id))
        if "file" in sina_kargs and "file_uri" in sina_kargs:
            raise ValueError(
                "The `file` keyword is being deprecated for `file_uri` you cannot use both")
        if "file" in sina_kargs:
            warnings.warn(
                "The `file` keyword has been renamed `file_uri` and may not be available in future versions",
                DeprecationWarning)
            file_uri = sina_kargs.pop("file")
            sina_kargs["file_uri"] = file_uri

        # The data dict for sina
        keys.pop("id_pool", None)
        sina_kargs["query_order"] = keys.pop(
            "query_order", ("data", "file_uri", "types"))
        sina_data = keys.pop("data", {})
        for att in atts:
            sina_data[att] = sina.utils.exists()
        sina_data.update(keys)
        sina_kargs["data"] = sina_data

        match = set(
            self.__record_handler__.find(
                id_pool=inter_recs,
                **sina_kargs))
        # instantly restrict to associated data
        if not self.__store__.__sync__:
            match_mem = set(self.__store__._added_unsync_mem_store.records.find(
                id_pool=inter_recs, **sina_kargs))
            match = match.union(match_mem)
        # ok now we need to search the data on the virtual datasets
        rec = self.get_record()
        for uri in virtuals:
            file_section = rec["files"][uri]
            tags = file_section.get("tags", [])
            # Now let's search the tags....
            match_it = True
            for key in sina_kargs["data"]:
                if key == "mime_type":
                    if file_section.get("mimetype", file_section.get(
                            "mime_type", None)) != sina_kargs["data"][key]:
                        match_it = False
                        break
                elif key not in tags or sina_kargs["data"][key] != sina.utils.exists():
                    match_it = False
                    break
            if match_it:
                # we can't have a set anymore
                match.add(self.id)

        for rec_id in match:
            # We need to cleanup for "virtual association".
            # e.g comes directly from a sina rec with 'file'/'mimetype' in it.
            rec_id = rec_id.split("__uri__")[0]
            yield rec_id if ids_only else self.__store__._load(rec_id)

    def export(self, file=None):
        """Exports this dataset
        :param file: export dataset to a file
        :type file: None or str
        :return: dataset and its associated data
        :rtype: dict"""
        rec = self.get_record()
        # cleanup the record
        rec_json = cleanup_sina_record_from_kosh_sync(rec)
        jsns = [rec_json, ]
        # ok now same for associated data
        for associated_id in self._associated_data_:
            rec = self.__store__._load(associated_id).get_record()
            rec_json = cleanup_sina_record_from_kosh_sync(rec)
            jsns.append(rec_json)

        # returns a dict that should be ingestible by sina
        output_dict = {
            "minimum_kosh_version": None,
            "kosh_version": kosh.version(comparable=True),
            "sources_type": self.__store__._sources_type,
            "records": jsns
        }

        update_json_file_with_records_and_relationships(file, output_dict)
        return output_dict

    def get_associated_data(self, ids_only=False):
        """Generator of associated data
        :param ids_only: generator will return ids if True Kosh object otherwise
        :type ids_only: bool
        :returns: generator
        :rtype: str or Kosh objects
        """
        for id in self._associated_data_:
            if ids_only:
                yield id
            else:
                yield self.__store__._load(id)

    def is_member_of(self, ensemble):
        """Determines if this dataset is a member of the passed ensemble
        :param ensemble: ensemble we need to determine if this dataset is part of
        :type ensemble: str or KoshEnsemble

        :returns: True if member of the ensemble, False otherwise
        :rtype: bool"""
        if not isinstance(ensemble, (basestring, kosh.ensemble.KoshEnsemble)):
            raise TypeError("ensemble must be id or KoshEnsemble object")
        if isinstance(ensemble, kosh.ensemble.KoshEnsemble):
            ensemble = ensemble.id

        return ensemble in self.get_ensembles(ids_only=True)

    def get_ensembles(self, ids_only=False):
        """Returns the ensembles this dataset is part of
        :param ids_only: return ids or objects
        :type ids_only: bool
        """
        for rel in self.get_sina_store().relationships.find(self.id, self.__store__._ensemble_predicate, None):
            if ids_only:
                yield rel.object_id
            else:
                yield self.__store__.open(rel.object_id)

    def leave_ensemble(self, ensemble):
        """Removes this dataset from an ensemble
        :param ensemble: The ensemble to leave
        :type ensemble: str or KoshEnsemble
        """
        from kosh.ensemble import KoshEnsemble
        if isinstance(ensemble, basestring):
            ensemble = self.__store__.open(ensemble)
        if not isinstance(ensemble, KoshEnsemble):
            raise ValueError("cannot join `ensemble` since object `{}` does not map to an ensemble".format(ensemble))
        if self.id in ensemble.get_members(ids_only=True):
            ensemble.remove(self.id)
        else:
            warnings.warn("{} is not part of ensemble {}. Ignoring request to leave it.".format(self.id, ensemble.id))

    def join_ensemble(self, ensemble):
        """Adds this dataset to an ensemble
        :param ensemble: The ensemble to join
        :type ensemble: str or KoshEnsemble
        """
        from kosh.ensemble import KoshEnsemble
        if isinstance(ensemble, basestring):
            ensemble = self.__store__.open(ensemble)
        if not isinstance(ensemble, KoshEnsemble):
            raise ValueError("cannot join `ensebmle` since object `{}` does not map to an ensemble".format(ensemble))
        ensemble.add(self)

    def clone(self, preserve_ensembles_memberships=False, id_only=False):
        """Clones the dataset, e.g makes an identical copy.

        :param preserve_ensembles_memberships: Add the new dataset to the ensembles this dataset belongs to?
        :type preserve_ensembles_membership: bool

        :param id_only: returns id rather than new dataset
        :type id_only: bool

        :returns: The cloned dataset or its id
        :rtype: KoshDataset or str
        """
        attributes = self.list_attributes(True)
        cloned_dataset = self.__store__.create(metadata=attributes)
        for associated in self.get_associated_data():
            cloned_dataset.associate(associated.uri, associated.mime_type, metadata=associated.list_attributes(True))

        if preserve_ensembles_memberships:
            for ensemble in self.get_ensembles():
                ensemble.add(cloned_dataset)

        if id_only:
            return cloned_dataset.id
        else:
            return cloned_dataset
