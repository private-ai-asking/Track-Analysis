from typing import List

from track_analysis.components.md_common_python.py_common.patterns import IPipe
from track_analysis.components.track_analysis.features.data_generation.model.album_cost import AlbumCostModel
from track_analysis.components.track_analysis.features.data_generation.pipeline.pipeline_context import PipelineContextModel


class GetAlbumCosts(IPipe):
    def _add_album_cost(self, album_costs: List[AlbumCostModel], title: str, cost: float) -> List[AlbumCostModel]:
        album_costs.append(AlbumCostModel(Album_Title=title, Album_Cost=cost))
        return album_costs

    def flow(self, data: PipelineContextModel) -> PipelineContextModel:
        album_costs = []

        album_costs = self._add_album_cost(album_costs, "Classical Best", 10.49)
        album_costs = self._add_album_cost(album_costs, "The Hours (Music from the Motion Picture)", 12.49)
        album_costs = self._add_album_cost(album_costs, "Old Friends New Friends", 20.39)

        album_costs = self._add_album_cost(album_costs, "Musica baltica", 15.19)
        album_costs = self._add_album_cost(album_costs, "Prehension", 16.29)
        album_costs = self._add_album_cost(album_costs, "Solipsism", 13.59)

        album_costs = self._add_album_cost(album_costs, "The Blue Notebooks (20 Year Edition)", 16.29)
        album_costs = self._add_album_cost(album_costs, "In a Time Lapse", 16.29)
        album_costs = self._add_album_cost(album_costs, "Una mattina", 16.29)
        album_costs = self._add_album_cost(album_costs, "Eden Roc", 16.29)
        album_costs = self._add_album_cost(album_costs, "I Giorni", 16.29)
        album_costs = self._add_album_cost(album_costs, "Le onde", 16.29)

        album_costs = self._add_album_cost(album_costs, "Lead Thou Me On: Hymns and Inspiration", 9.49)
        album_costs = self._add_album_cost(album_costs, "Lux", 13.59)
        album_costs = self._add_album_cost(album_costs, "Eventide", 13.59)
        album_costs = self._add_album_cost(album_costs, "Light and Gold", 30.79)
        album_costs = self._add_album_cost(album_costs, "De la taberna a la Corte", 12.59)
        album_costs = self._add_album_cost(album_costs, "Edvard Grieg a capella", 10.49)
        album_costs = self._add_album_cost(album_costs, "Edvard Grieg - Essential Orchestral Works", 5.79)
        album_costs = self._add_album_cost(album_costs, "The Young Beethoven", 10.79)
        album_costs = self._add_album_cost(album_costs, "The Young Messiah", 10.79)
        album_costs = self._add_album_cost(album_costs, "Ode To Joy", 10.79)
        album_costs = self._add_album_cost(album_costs, "Satie: Gymnopédies; Gnossiennes", 8.59)
        album_costs = self._add_album_cost(album_costs, "The Very Best of Arvo Pärt", 9.29)
        album_costs = self._add_album_cost(album_costs, "Elegy for the Arctic", 1.99)
        album_costs = self._add_album_cost(album_costs, "Alina", 13.59)
        album_costs = self._add_album_cost(album_costs, "Divenire", 16.29)
        album_costs = self._add_album_cost(album_costs, "Elements", 20.69)
        album_costs = self._add_album_cost(album_costs, "Memoryhouse", 10.79)

        album_costs = self._add_album_cost(album_costs, "Halfway Tree", 13.59)
        album_costs = self._add_album_cost(album_costs, "Welcome to Jamrock", 13.59)
        album_costs = self._add_album_cost(album_costs, "Mr. Marley", 13.59)
        album_costs = self._add_album_cost(album_costs, "Distant Relatives", 13.59)
        album_costs = self._add_album_cost(album_costs, "Rapture", 6.49)
        album_costs = self._add_album_cost(album_costs, "Stony Hill", 23.99)
        album_costs = self._add_album_cost(album_costs, "A Matter of Time", 8.99)

        data.album_costs = album_costs
        return data
