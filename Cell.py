from scipy.spatial import distance


class Cell(object):
    """Provides an object to wrap cell data
    Attributes:
        x                x-axis location of a cell (um)
        y                y-axis location of a cell (um)
        activities       dictionary {marker: boolean} of marker boolean activities
        scores           dictionary {marker: float} of marker continuous activity values
        tissue_type      tissue type to which the cell belongs [Cancer, Stroma]
        phenotype_label  string of all active marker names in cell separated by dash (-)
        on_margin       boolean - does
        margin_edge
        margin_number    int - id number of margin to which the cell belongs

    """

    def __init__(self, record, idx):
        """
        Parameters
        ----------
        record : DataFrame
            DataFrame containing one record of description of IF panel cell

        Returns
        -------
        Cell
            Cell object with that includes:
                x -
                y -
        """
        self.id = idx
        self.x = record['nucleus.x']
        self.y = record['nucleus.y']
        self.activities = dict(
            [(i.split(".")[0], float(record[i]) > 1) for i in record.keys() if "score.normalized" in i])
        self.scores = dict([(i.split(".")[0], float(record[i])) for i in record.keys() if "score.normalized" in i])
        self.tissue_type = record['tissue.type']
        _phenotypes = [key for key, value in self.activities.items() if value]

        self.phenotype_label = '-'.join(sorted(_phenotypes, reverse=True))
        if self.phenotype_label == "":
            self.phenotype_label = "neg"

        self.on_margin = False
        self.margin_edge = None
        self.margin_number = -1

    def __eq__(self, other):
        """Compares two cells"""
        if isinstance(other, Cell):
            return self.x == other.x and self.y == other.y and self.phenotype_label == other.phenotype_label
        return False

    def __hash__(self):
        return hash(str(self))

    @property
    def position(self):
        """The radius property."""
        return self.x, self.y

    def __delitem__(self, key):
        self.activities.__delattr__(key)

    def __getitem__(self, key):
        return self.activities.__getattribute__(key)

    def __setitem__(self, key, value):
        self.activities.__setattr__(key, value)

    def distance(self, other_cell):
        """Calculates euclidean distance between from the other_cell
        Parameters
        ----------
        other_cell : Cell
            a cell to which a distance needs to be calculated

        Returns
        -------
        distance
            distance from the cell to other_cell
        """
        return distance.euclidean((self.x, self.y), (other_cell.x, other_cell.y))

    def __str__(self):
        res = ""
        res += self.phenotype_label + " "
        res += "at: "
        res += "(" + str(self.x) + ", " + str(self.y) + ")"
        return res
