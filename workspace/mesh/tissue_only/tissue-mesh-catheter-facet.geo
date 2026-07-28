SetFactory("OpenCASCADE");
Merge "tissue-mesh-catheter-facet.step";
Mesh.CharacteristicLengthMax = 1;
//+
Disk(7) = {0, 0, 0, 1.25, 1.25};
//+
BooleanFragments{ Volume{1}; Delete; }{ Surface{7}; Delete; }
//+
Physical Surface("air_interface", 26) = {10};
//+
Physical Surface("sides", 27) = {9, 8, 12, 11};
//+
Physical Surface("catheter", 28) = {7};
//+
Physical Surface("base", 29) = {13};
//+
Physical Volume("tissue_volume", 30) = {1};
//+
// Refine mesh around the lesion site.
Field[1] = Distance;
Field[1].SurfacesList = {7}; 
Field[2] = Threshold;
Field[2].InField = 1;      // Link to the distance field above
Field[2].SizeMin = 0.1;    // The fine mesh size on and immediately around the fragment
Field[2].SizeMax = 1.0;    // The coarse mesh size far away from the fragment
Field[2].DistMin = 0.2;    // Distance from the surface where SizeMin strictly applies
Field[2].DistMax = 1.5;    // Distance at which the mesh reaches SizeMax
Background Field = 2;
Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh(3);
