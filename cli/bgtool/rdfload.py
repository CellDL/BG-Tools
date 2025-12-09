class BondgraphModelSource:
    def __init__(self, source: Path|str, output_rdf: Optional[Path]=None, debug=False):
        self.__rdf_graph = RDFGraph(NAMESPACES)
        self.__source_path = Path(source).resolve()
        self.__loaded_sources: set[Path] = set()
        self.__load_rdf(self.__source_path)
        base_models = []
        for row in self.__rdf_graph.query(BONDGRAPH_MODELS):
            uri = cast(NamedNode, row[0])
            base_models.append((uri, row[1]))
            self.__load_blocks(uri)
            self.__generate_bonds(uri)

        if len(base_models) < 1:
            log.error(f'No BondgraphModels in source {source}')

        if output_rdf is not None:
            with open(output_rdf, 'w') as fp:
                fp.write(self.__rdf_graph.serialise(source_url=self.__source_path.as_uri()))
            log.info(f'Expanded model saved as {pretty_log(output_rdf)}')

        self.__models: dict[NamedNode, BondgraphModel] = {}
        for (uri, label) in base_models:
            for row in self.__rdf_graph.query(BONDGRAPH_MODEL_TEMPLATES.replace('%MODEL%', uri)):
                self.__add_template(row[0])
            self.__models[uri] = BondgraphModel(self.__rdf_graph, uri, label, debug=debug)

    @property
    def models(self):
        return list(self.__models.values())

    def __add_template(self, path: ResultType):
    #==========================================
        if isNamedNode(path):
            FRAMEWORK.add_template(path)    # pyright: ignore[reportArgumentType]
        elif isLiteral(path):
            FRAMEWORK.add_template(self.__source_path
                                        .parent
                                        .joinpath(str(path))
                                        .resolve())

    def __generate_bonds(self, model_uri: NamedNode):
    #=============================================
        for row in self.__rdf_graph.query(BONDGRAPH_BONDS):
            if isNamedNode(row[1]):
                source = row[1]
            elif isBlankNode(row[1]) and isNamedNode(row[3]):
                source = row[3]
            else:
                source = None
            if isNamedNode(row[2]):
                target = row[2]
            elif isBlankNode(row[2]) and isNamedNode(row[4]):
                target = row[4]
            else:
                target = None
            if ((Triple(None, BGF.hasBondElement, source) in self.__rdf_graph
              or Triple(None, BGF.hasJunctionStructure, source) in self.__rdf_graph)
            and (Triple(None, BGF.hasBondElement, target) in self.__rdf_graph
              or Triple(None, BGF.hasJunctionStructure, target) in self.__rdf_graph)):
                self.__rdf_graph.add(Triple(model_uri, BGF.hasPowerBond, row[0]))

    def __load_blocks(self, model_uri: NamedNode):
    #=============================================
        ## need to make sure blocks are only loaded once. c.f templates
        for row in self.__rdf_graph.query(BONDGRAPH_MODEL_BLOCKS.replace('%MODEL%', model_uri.value)):
            self.__load_rdf(Path.from_uri(urldefrag(str(row[0])).url))

    def __load_rdf(self, source_path: Path):
    #=======================================
        if source_path not in self.__loaded_sources:
            self.__loaded_sources.add(source_path)
            graph = RDFGraph(NAMESPACES)
            graph.parse(source_path)
            for row in graph.query(BONDGRAPH_MODELS):
                if isNamedNode(row[0]):
                    #FRAMEWORK.resolve_composites(row[0], graph)
                    self.__load_blocks(row[0])      # pyright: ignore[reportArgumentType]
            self.__rdf_graph.merge(graph)
