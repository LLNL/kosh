import numpy
from .core import KoshLoader


class SidreMeshBlueprintFieldLoader(KoshLoader):
    types = {"sidre_mesh_blueprint_fields":
             ["numpy", "sidre/path"]}

    def extract(self, *args, **kargs):
        import conduit
        import conduit.relay
        if not isinstance(self.feature, list):
            features = [self.feature]
        else:
            features = self.feature

        ioh = conduit.relay.io.IOHandle()
        ioh.open(self.uri, "sidre_hdf5")
        # look for self.feature
        bp_idx = conduit.Node()
        ioh.read(bp_idx, "root/blueprint_index")
        bp_path = conduit.Node()
        ndoms = conduit.Node()
        ioh.read(bp_path, "root/blueprint_index")

        out = []
        for feature in features:
            # split feature name back to mesh + field
            sp = feature.split("/")
            mesh = sp[0]
            field = "/".join(sp[1:])
            # get number of domains for this mesh
            ioh.read(
                ndoms,
                "root/blueprint_index/{}/state/number_of_domains".format(mesh))
            ndoms = ndoms.value()
            # get the path to the selected field in the bulk data
            pth = "root/blueprint_index/{}/fields/{}/path".format(mesh, field)
            if self.format == "sidre/path":
                out.append([ioh, pth])
                continue
            ioh.read(bp_path, pth)
            bp_path = bp_path.value()
            res = None
            vals = conduit.Node()
            for i in range(ndoms):
                dom_path = "%d/" % i
                dom_path += bp_path + "/values"
                ioh.read(vals, dom_path)
                npy_array = vals.value()
                if res is not None:
                    res = numpy.concatenate([res, npy_array])
                else:
                    res = npy_array
            out.append(res)
        if len(features) == 1:
            return out[0]
        else:
            return out

    def list_features(self):
        import conduit
        import conduit.relay
        ioh = conduit.relay.io.IOHandle()
        ioh.open(self.uri, "sidre_hdf5")
        # get the blueprint index
        bp_idx = conduit.Node()
        ioh.read(bp_idx, "root/blueprint_index")
        # enumerate meshes, and fields on each mesh
        res = []
        for mesh in bp_idx.children():
            for field in mesh.node()["fields"].children():
                res.append("/".join([mesh.name(), field.name()]))
        return res
