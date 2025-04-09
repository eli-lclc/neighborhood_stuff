import pandas as pd
import os
import numpy as np
import re

from pykml import parser
from shapely.geometry import Point, Polygon
import geopandas as gpd
from geopandas.tools import geocode
from geopy.geocoders import GoogleV3
from shapely import wkt

import plotly.graph_objects as go
import plotly.express as px


class Add_Neighborhoods():
    '''
    Class for assigning neighborhoods to a dataframe of addresses.

    Parameters:

        maps_api_key: your google maps api key
        address_df: dataframe of addresses you wish to add neighborhoods to
        existing_file: file path to a df (either csv or exel) of addresses with neighborhoods to merge with address_df.
        region_kml: kml file path for broader regions outside of desired neighborhoods. defaults to lclc_regions.kml
        neighborhood_kml: kml file path of neighborhood boundaries. defaults to Chicago Neighborhoods.kml
        id_col: name of the df's id column. defaults to 'address_id', set to None if no column exists
        address1_col: name of the df's address1 column. Defaults to 'address1'
        city_col: name of the df's 'city' column. Defaults to 'city', set to None if no column exists
        state_col: name of the df's 'state' col. Defaults to 'state', set to None if no col exists
        zip_col: name of df's 'zip' col.
    '''
    def __init__(self, maps_api_key, address_df,existing_file, region_kml, 
    neighborhood_kml, id_col, address1_col, city_col, state_col, zip_col):
        self.address_df = address_df
        self.maps_api_key = maps_api_key
        self.address1_col = address1_col
        self.id_col = id_col if id_col else self.add_column('id')
        self.city_col = city_col if city_col else self.add_column('city')
        self.state_col = state_col if state_col else self.add_column('state') 
        self.zip_col = zip_col

        self.address_df[self.zip_col] = self.address_df[self.zip_col].astype(str)
        

        self.existing_file_path = existing_file
        if existing_file == None:
            self.existing_file = None
        elif existing_file.lower().endswith('csv'):
            self.existing_file = pd.read_csv(existing_file)
        else:
            self.existing_file = pd.read_excel(existing_file)
        
        self.full_neighborhoods = gpd.read_file(neighborhood_kml)
        self.neighborhood_boundaries = self.read_kml(neighborhood_kml)
        self.region_boundaries = self.read_kml(region_kml)

        
    def add_column(self, column):
        '''fills in missing/blank columns'''
        if column == 'id':
            self.id_col = 'address_id'
            self.address_df['address_id'] = df.reset_index().index
        if column == 'city':
            self.city_col = 'city'
            self.address_df['city'] = 'Chicago'
        elif column == 'state':
            self.state_col ='state'
            self.address_df['state'] = 'Illinois'
        return column

    def read_kml(self, file_path):
        ''' reads the KML '''
        with open(file_path, 'r') as f:
            root = parser.parse(f).getroot()

        boundaries = {}
        for placemark in root.Document.iterdescendants(tag='{http://www.opengis.net/kml/2.2}Placemark'):
            name = placemark.name.text
            coordinates_str = placemark.Polygon.outerBoundaryIs.LinearRing.coordinates.text.strip()
            coordinates = [tuple(map(float, coord.split(','))) for coord in coordinates_str.split()]
            polygon = Polygon(coordinates)
            boundaries[name] = polygon

        return boundaries

    def add_neighborhoods(self):
        def find_boundary(point, boundaries):
            for boundary_name, boundary_polygon in boundaries.items():
                if point.within(boundary_polygon):
                    return boundary_name
            return 'Other_Suburbs'

        def clean_address(address):
            address = re.sub(r'\s+', ' ', address)  # Remove extra whitespaces
            address = re.sub(r'[^\w\s,]', '', address)  # Remove special characters
            address = address.strip()  # Remove leading/trailing spaces
            return address

        def construct_full_address(row):
            parts = [
                row[self.address1_col].strip(),
                row[self.city_col].strip(),
                row[self.state_col].strip() + ' ' + row[self.zip_col].strip() if row[self.zip_col].strip() else row[self.state_col].strip()
            ]
            return ', '.join([part for part in parts if part])

        def check_if_point_col_exists(df):
            '''check if POINT() column exists in df'''
            threshold = len(df)/2
            for col in df.columns:
                num_point_rows = df[col].astype(str).str.startswith("POINT (").sum()
                if num_point_rows >= threshold:
                    return col
            return None
        
        point_col = check_if_point_col_exists(self.address_df)
        
        #check if DF has a location column already, rarely the case
        if point_col:
            self.address_df = self.address_df.rename(columns={point_col:'geometry'})
            gdf_addresses = self.address_df

            # create columns that are added otherwise
            gdf_addresses['location'] = gdf_addresses['geometry']
            gdf_addresses['full_address'] = gdf_addresses[self.address1_col]
            gdf_addresses['geometry'] = gdf_addresses['geometry'].apply(wkt.loads)
            #convert to gdf
            gdf_addresses = gpd.GeoDataFrame(gdf_addresses, geometry='geometry')
        
        else:
            geolocator = GoogleV3(api_key=self.maps_api_key)
            #merges with existing CSV as applicable, removing rows with existing neighborhood assignments
            if self.existing_file is not None:
                self.address_df = self.address_df[~self.address_df.loc[:,self.id_col].isin(self.existing_file[self.id_col])]
                if len(self.address_df) == 0:
                    self.neighborhood_df = self.existing_file
                    return self.existing_file
            
            #fills blank values
            self.address_df.loc[~self.address_df[self.address1_col].isnull(), self.city_col] = self.address_df[self.city_col].fillna('Chicago')
            self.address_df.loc[~self.address_df[self.address1_col].isnull(), self.state_col] = self.address_df[self.state_col].fillna('Illinois')
            self.address_df[[self.address1_col, self.city_col, self.state_col, self.zip_col]] = self.address_df[[self.address1_col, self.city_col, self.state_col, self.zip_col]].fillna('')
            
            # construct/clean full_address column
            self.address_df['full_address'] = self.address_df.apply(lambda row: construct_full_address(row), axis=1)
            self.address_df['full_address']=self.address_df['full_address'].apply(clean_address)

            # finds coordinates
            self.address_df['location'] = self.address_df['full_address'].apply(lambda x: geolocator.geocode(x, timeout=10) if x else None)
            
            # makes dataframe of all non-null locations
            gdf_addresses = gpd.GeoDataFrame(self.address_df[self.address_df['location'].notnull()],
                                        geometry=self.address_df['location'].apply(lambda loc: Point(loc.longitude, loc.latitude) if loc else None))
        
        # attaches neighborhood to coordinates
        gdf_addresses['neighborhood'] = gdf_addresses.geometry.apply(lambda geom: find_boundary(geom, self.neighborhood_boundaries))


        condition = (gdf_addresses['neighborhood'] == 'Other_Suburbs') & (gdf_addresses[self.city_col] == 'Chicago')
        # checks region map for matches (ie: if an address is in chicago but not a selected neighborhood from neighborhood.kml)
        gdf_addresses.loc[condition, 'neighborhood'] = gdf_addresses.loc[condition].geometry.apply(lambda geom: find_boundary(geom, self.region_boundaries))
        gdf_addresses.loc[(gdf_addresses[self.city_col] == 'Chicago') & (gdf_addresses['neighborhood'] == 'Other_Suburbs'), 'neighborhood'] = 'Other_Chicago'
        gdf_addresses.loc[(gdf_addresses['location'] == 'Nanno, Ville d''Anaunia, Comunità della Val di Non, Provincia di Trento, Trentino-Alto Adige/Südtirol, 38093, Italia'), 'neighborhood'] = 'Other_Missing'

        #print('pog')

        self.neighborhood_df = gdf_addresses

        # merge with existing excel/csv file if applicable
        if self.existing_file is not None:
            self.standardize_gdf()
            self.neighborhood_df = pd.concat([self.existing_file,self.neighborhood_df])
        return self.neighborhood_df

    def standardize_gdf(self):
        #s standardizes column names
        self.neighborhood_df = self.neighborhood_df[[self.id_col, self.address1_col,self.city_col,self.state_col,self.zip_col,'full_address','location','geometry','neighborhood']]

    def push_to_csv(self, csv_path = None):
        self.standardize_gdf()
        if self.existing_file is None:
            self.neighborhood_df.to_csv(csv_path, encoding='utf-8', index=False)
        else:
            pd.concat([self.existing_file,self.neighborhood_df]).to_csv(self.existing_file_path, encoding='utf-8', index=False)

class Neighborhoods(Add_Neighborhoods):
    '''
    Creates an object to store information on the dataframe that needs neighborhoods added

    Parameters:
        maps_api_key: your google maps api key
        address_df: dataframe of addresses you wish to add neighborhoods to
        existing_file: file path to a df (either csv or exel) of addresses with neighborhoods. 
        add_neighborhoods (Bool): whether neighborhoods need to be added to the DF. defaults to False
        region_kml: kml file path for broader regions outside of desired neighborhoods. defaults to lclc_regions.kml
        neighborhood_kml: kml file path of neighborhood boundaries. defaults to Chicago Neighborhoods.kml
        id_col: name of the df's id column. defaults to 'address_id', set to None if no column exists
        address1_col: name of the df's address1 column. Defaults to 'address1'
        city_col: name of the df's 'city' column. Defaults to 'city', set to None if no column exists
        state_col: name of the df's 'state' col. Defaults to 'state', set to None if no col exists
        zip_col: name of df's 'zip' col.
    '''
    def __init__(self, maps_api_key, address_df, existing_file = None, add_neighborhoods = False,
        region_kml = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sample_docs', 'lclc_regions.kml'), 
        neighborhood_kml =os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sample_docs', 'Chicago Neighborhoods.kml'),
        id_col='address_id', address1_col ='address1', city_col='city', state_col='state', zip_col='zip'):
        super(Neighborhoods, self).__init__(maps_api_key, address_df, existing_file, region_kml, neighborhood_kml, id_col, address1_col, city_col, state_col, zip_col)
        if add_neighborhoods:
            self.add_neighborhoods()
            self.standardize_gdf()
        else:
            self.neighborhood_df = address_df

#graphing garbage, unsure of how to sort
    def create_colorscale(self, colorscale, neighbor_df, five_or_fewer = False):
        def decrement_rgb(rgb_string):
            # Extract the numbers from the string
            rgb_values = list(map(int, re.findall(r'\d+', rgb_string)))
            # Subtract 1 from each value, ensuring they don't go below 0
            new_rgb_values = [max(0, v - 1) for v in rgb_values]
            # Format back into an RGB string
            return f"rgb({new_rgb_values[0]},{new_rgb_values[1]},{new_rgb_values[2]})"

        colorscale_temp = colorscale
        max_val = (neighbor_df['count'].max())
        color_for_0 = 1/(max_val+1)
        interval = max_val/((max_val+1)*(len(colorscale_temp)-2))
        color_thingy = []
        if five_or_fewer:
            # adjust interval to account for a decreased range
            interval = (max_val-5) /((max_val+1)*(len(colorscale_temp)-2))

        for i, rgb in enumerate(colorscale_temp):
            if i == 0:
                new_val = 0
            
            elif five_or_fewer:
                if i== 1:
                    new_val = color_for_0 + (4.5/max_val)
                    pct_at_1 = 1/max_val
                    color_thingy.append([pct_at_1,rgb])
                    # change color slightly so continuous scale still works
                    rgb = decrement_rgb(rgb)
                    pass
                elif i == 2:
                    new_val = color_for_0 + 5/max_val
                else:
                    new_val = color_for_0 + interval * (i-1)
            
            else:
                new_val = color_for_0 + interval * (i-1)
            color_thingy.append([new_val, rgb])
        color_thingy[-1][0] = 1
        return(color_thingy)

    # Determine text color based on luminance
    def determine_text_color(self, percentag):
        '''makes text white if region color will be dark'''

        def calculate_luminance(rgb_color):
            rgb_values_str = rgb_color[4:-1]  # Remove 'rgb(' and ')'
            rgb_values = [float(colour) for colour in rgb_values_str.split(',')]
            r, g, b = rgb_values
            return 0.2126*r + 0.7152*g + 0.0722*b

        for color_boundary, color in self.custom_colorscale:
            if percentag >= color_boundary:
                percentag_color = color
            else:
                break
        luminance = calculate_luminance(percentag_color)
        return 'white' if luminance < 128 else 'black'
    
    def format_title_text(self, title_str):
        '''adds a line break if title is real long'''
        if len(title_str) > 30:
            n = len(title_str) // 2
            second_line = title_str[n:]
            first_space_index = second_line.find(' ')

            line_br = n + first_space_index + 1
            first_line = title_str[:line_br]
            second_line = title_str[line_br:]
            result = f"{first_line}<br>{second_line}"
            return result
        else:
            return title_str



    def compute_area_percentile(self, df):

        '''Determines how big a neighborhood's area is compared to other areas'''

        df["area"] = df["geometry"].area  

        # Compute min and max area
        min_area = df["area"].min()
        max_area = df["area"].max()

        if min_area == max_area:
            df["area_percentile"] = 1.0
        else:
            df["area_percentile"] = (df["area"] - min_area) / (max_area - min_area)

        return df


    def prep_neighborhood_labels(self, df, min_font, max_font):
        '''
        Wrapper function. Adds shape category, splits the string as desired, 
        generates font size depending on minimum/maximum fonts
        '''
        def categorize_shape(row):
            '''Categorize the shape of a geometry as 'Tall', 'Wide', or 'Squareish'.'''
            minx, miny, maxx, maxy = row.geometry.bounds  # Get bounding box
            width = maxx - minx
            height = maxy - miny

            if height > width * 1.25:
                return 'Tall'
            elif width > height * 1.6:
                return 'Extra Wide'
            elif width > height * 1.25:
                return 'Wide'
            else:
                return 'Squareish'


        def split_string_evenly(str_0):
            ''' function to split string by closest space'''

            mid_point = int(len(str_0)/2)
            if str_0[mid_point]==' ':
                # the middle point is a space
                str_1 = str_0[:mid_point]
                str_2 = str_0[mid_point+1:]
                return (f'{str_1}<br>{str_2}')

            for i in range(1,mid_point):
                # the middle point is a character
                if str_0[mid_point-i] == ' ':
                    break_point = mid_point-i
                    break

                if str_0[mid_point+i] == ' ':
                    break_point = mid_point+i
                    break

            str_1 = str_0[:break_point]
            str_2 = str_0[break_point+1:]
            return (f'{str_1}<br>{str_2}')

        def split_string_fully(str_0):
            '''split string at every space'''
            return ("<br>".join(str_0.split()))

        def generate_label_str(i, row):
            '''determines how much label splitting to do depending on row's shape and label length'''
            
            if len(i.split()) == 1 or ( 'Wide'  in row.shape_category and len(i) < 15):
                return i
            elif row.shape_category == 'Squareish' or 'Wide' in row.shape_category:
                return split_string_evenly(i)
            else:
                return split_string_fully(i)

        def generate_label_size(row, min_font, max_font):
            label_size = round(min_font + (max_font-min_font)*row.area_percentile)
            return label_size

        ### actual wrapping
        df = self.compute_area_percentile(df)
        df["shape_category"] = df.apply(categorize_shape, axis=1)
        df['neighborhood_label'] = df.apply(lambda row: generate_label_str(row.name, row), axis=1)
        df['label_size'] = df.apply(lambda row: generate_label_size(row,min_font,max_font), axis=1)
        return df
    
    def get_colorbar_scale(self, max_val, five_or_fewer, potential_tickscales):
        '''Evenly distributes the colors across the range of counts'''
        scale_value = min(potential_tickscales, key=lambda x:abs(x-(max_val/5)))

        range_values = [int(x) for x in list(range(0, int(max_val), int(scale_value)))]
        
        range_labels = [str(x) for x in range_values]

        if five_or_fewer:
            range_values[0]=3
            range_labels[0]='Five or fewer'

        return range_values, range_labels


    def heat_map(self,chart_title = None, kml_nieghborhood_col = 'Name', color_bar_label = 'Count', five_or_fewer = False, 
        color_scale_sequence = px.colors.sequential.PuBu, potential_tickscales = [2,5,10,25,50,75,100]):
        '''
        Creates a heat map of neighborhoods

        Parameters:
            chart_title: desired chart title. defaults to None
            kml_neighborhood_col: column in the neighborhood kml df that denotes neighborhoods. defaults to 'Name'
            color_bar_label: label for the color bar. defaults to 'Count'
            five_or_fewer (Bool): True removes value labels for Neighborhoods where N <=5. Defaults to False
            color_scale_squence: color gradient for the heat map. defaults to px.colors.sequential.PuBu from plotly
            potential_tickscales: range of options for colorbar tick scale. Defaults to [2,5,10,25,50,75,100]
        '''
        

        
        #merge address df with the neighborhood locations df
        all_neighborhoods = self.full_neighborhoods.merge((self.neighborhood_df.groupby('neighborhood').size().reset_index(name='count')), 
        how='left', left_on=kml_nieghborhood_col, right_on='neighborhood')
        all_neighborhoods['count'] = all_neighborhoods['count'].fillna(0)
        all_neighborhoods = all_neighborhoods.set_index('Name')
        self.all_neighborhoods = all_neighborhoods
        
        #generate color scale
        self.custom_colorscale = self.create_colorscale(color_scale_sequence, all_neighborhoods,five_or_fewer)
        
        opacity_array = all_neighborhoods['count'].apply(lambda x: 0.4 if x == 0 else .8).tolist()
        
        # set up labels for neighborhoods
        labels_df = all_neighborhoods[all_neighborhoods['count'] > 0]
        
        max_val = labels_df['count'].max()

        labels_df['text_color'] = (labels_df['count']/max_val).apply(self.determine_text_color)
        labels_df['count']= pd.to_numeric(labels_df['count'], errors='coerce').fillna(0).astype(np.int64)
        labels_df = labels_df.set_index('neighborhood')
        labels_df = self.prep_neighborhood_labels(labels_df, 11, 20)
        self.labels_df=labels_df
        

        tickvals, ticktext = self.get_colorbar_scale(max_val, five_or_fewer, potential_tickscales)
        
        ### ACTUAL FIGURE PART
        fig = go.Figure()
        
        # base map
        fig.add_trace(go.Choroplethmapbox(
            geojson=all_neighborhoods.geometry.__geo_interface__,
            locations=all_neighborhoods.index,
            z=all_neighborhoods['count'],
            colorscale=self.custom_colorscale,
            zmin=0,
            zmax=all_neighborhoods['count'].max(),
            marker_opacity=opacity_array,  # Apply the custom opacity array
            marker_line_width=0.5,
            name = '',
            colorbar=dict(
                tickfont=dict(size=18,family= 'verdana'),
                title=color_bar_label,
                xanchor="right", x=.95,
                yanchor='top', y=1,
                thickness=30,
                xpad = 10,
                len=.5,
                tickvals = tickvals,
                ticktext = ticktext
            )
        ))
        # add chart title, subtitle
        if chart_title:
            fig.update_layout( title = dict(
                text=self.format_title_text(chart_title),font=dict(size=40, family= 'verdana'), automargin=True, yref='paper'),
            title_x = 0.5,
                annotations=[
                dict(font=dict(size=30, family='Verdana', color=color_scale_sequence[-2]),
                    text=self.format_title_text(f'(n={len(self.neighborhood_df)})'),
                    x=0.5,
                    y=1,
                    xref='paper',
                    yref='paper',
                    showarrow=False,
                    bgcolor="white")
    ])
        # add labels for neighborhoods
        for i, row in labels_df.iterrows():
            if five_or_fewer and row['count'] < 6:
                # only add neighborhood name
                fig.add_trace(go.Scattermapbox(
                    lat=[row.geometry.centroid.y],
                    lon=[row.geometry.centroid.x],
                    mode='markers+text',
                    text=[str(row['neighborhood_label'])],  # Row index as text
                    textfont=dict(color=row['text_color'], size=row['label_size']),
                    marker=go.scattermapbox.Marker(size=0),    # Small font for index
                    textposition='middle center',
                    hoverinfo='skip'
            ))

            else:
                # determine where to place name/number in a neighborhood depending on shape
                minx, miny, maxx, maxy = row.geometry.bounds  
                height = maxy - miny
                if "Extra Wide" in row['shape_category']:
                    top_y = maxy - 0.4 * height
                    bottom_y = miny + 0.35 * height
                else:
                    top_y = maxy - 0.4 * height
                    bottom_y = miny + 0.25 * height
                
                num_poaition = 'bottom right' if any(substring in i for substring in ["North Austin","Greater Grand Crossing"]) else 'middle center'
                text_position = 'middle left' if any(substring in i for substring in ["North Austin"]) else 'top center'
                
                # add neighborhood label
                fig.add_trace(go.Scattermapbox(
                    lat=[top_y],
                    lon=[row.geometry.centroid.x],
                    mode='markers+text',
                    marker=go.scattermapbox.Marker(size=0),
                    text=[str(row['neighborhood_label'])], 
                    textfont=dict(color=row['text_color'], size=row['label_size']),
                    textposition=text_position,
                    hoverinfo='skip'
            ))

                # add count
                fig.add_trace(go.Scattermapbox(
                    lat=[bottom_y],
                    lon=[row.geometry.centroid.x],
                    mode='markers+text',
                    marker=go.scattermapbox.Marker(size=0),  # No marker, only text
                    text=[str(row['count'])],
                    textfont=dict(color=row['text_color'], size=20),  # Main text font
                    textposition=num_poaition,
                    hoverinfo='skip'
                    
                ))
        
        # messing around with layout
        fig.update_layout(
            mapbox=dict(
                style="white-bg",
                zoom=11.5,
                center={"lat": labels_df.geometry.centroid.y.mean(), "lon": labels_df.geometry.centroid.x.mean()},
                bounds={"north": labels_df.geometry.centroid.y.max(), "south": labels_df.geometry.centroid.y.min()}

            ),
            margin={"r": 0, "t": 120, "l": 0, "b": 0},
            showlegend=False,
            autosize=False,
            width=1400,
            height=2100)

        return fig