import ee
ee.Initialize(project='sar-flood-anomaly-mapping')  # the project ID from when you registered for EE
print(ee.ImageCollection('COPERNICUS/S1_GRD').first().getInfo())