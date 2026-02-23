

need to create data outputs, for all standard shapes
* hexagon shaped hexagon only polygrids, with extent 1 through 12.
* pentagon centred polygrids, with extent 1 through 12.
* goldberg polyhedron polygrids, for frequencies 1 through 8.

Note - this data needs to be purely for faces and neighbours, everything we need to create a polygrid object that we can run algorithims on.
for visualation and testing:
* hexagon only grids can be calculated easily, to allow exporting to png.
* pentagon centred grids will need seperate data files output to support outputting pngs correctly of the polygrids.
* goldberg polygrids will make use of another library to visualise and validate - with no need to create .png files here.

so we need a png exporter for the 2d grids, and we should seperate that visualation logic from the core polygrid objects which are purely used for topology and algorithim runs. clear SoC here, in the code and in the data storage.