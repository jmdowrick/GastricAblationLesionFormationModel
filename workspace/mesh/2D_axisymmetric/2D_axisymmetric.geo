SetFactory("OpenCASCADE");
Merge "2D_axisymmetric.step";
Point(5) = {1.75, 0, 0, 1.0};
BooleanFragments{ Surface{1}; Delete; }{ Point{5}; Delete; }

// Define distance field
Field[1] = Distance;
Field[1].CurvesList = {1};
Field[1].NumPointsPerCurve = 100;

// Define a threshold field
Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = 0.1;
Field[2].SizeMax = 1.0;
Field[2].DistMin = 1;
Field[2].DistMax = 3.0;

Background Field = 2;

Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MeshSizeExtendFromBoundary = 0;

// Physical tags
Physical Surface("tissue", 6) = {1};
Physical Curve("catheter", 7) = {1};
Physical Curve("axis", 8) = {5};
Physical Curve("base", 9) = {4};
Physical Curve("tissue-outer", 10) = {3};
Physical Curve("air", 11) = {2};

// Make 2D mesh
Mesh(2);
RefineMesh;
RefineMesh;