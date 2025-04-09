Add neighborhoods to a dataframe of addresses! Make a glamorous heat map! The world is your oyster!

Clone this repository & install requirements.txt, and hopefully everything will work.

The code works by creating a Neighborhoods object, which is a child of the Add_Neighborhoods class. Add_Neighborhoods adds neighborhoods as needed and standardizes a dataframe (the attribute .neighborhood_df)

Once the object is initialized, one can run heat_map to create a heatmap using your df, a kml file of neighborhoods, and a kml file of regions. The defaults for KML files can be found in the sample_docs folder. 

heat_map creates a plotly Choropleth mapbox and utilizes a couple of custom functions to circumvent certain aesthetic yucks. Some of those functions:
    custom_colorscale: a function to evenly distribute the colors in the graph's colorscale
    determine_text_color: figures out a label's text color depending on a region's shade
    prep_neighborhood_labels: scales and splits a neighorbood name depending on the neighborhood's approximate shape and size

For a slick demo check out: tests/demo.ipynb