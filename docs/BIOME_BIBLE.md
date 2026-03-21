# Biome Bible

### Overview
** MOVE THIS INTO ORBIS SYLVA MAIN APP
This document will be used to talk about behaviours we want to encourage for various terrains and features via the procedural polygrid generation. 

### Biome Attribute Management

We will employ a tiered approach to reflect the system setup, with procedural generation only used on TileBiome level. Star, Planet and Region biome information will be from user input, or randomised (sometimes with influence from parent attributes).

#### StarBiome Attributes

* StarSize
* StarHeat (or intensity? Is this a thing in real life? Doesn't really matter if not I guess)

#### PlanetBiome Attributes

* Water (not sure how to phrase this best, Water Level? Water Abudance? Water?)
* Roughness (again not sure on best word here, but this will effect how irregular the planet is in terms of elevation)
* Distance from Star (along with star attributes, maybe also look at orbits and things like that, allows calculation of temperature)
* Temperature ()
* Minerals - List of minerals present, maybe grouped together to make simpler to manage? But will effect the look etc.


#### RegionBiome Attributes

These will be chosen by user, or randomised, utilising attributes from the parent.

* ElevationModifier
* HumidityModifier
* TemperatureModifier (maybe not this one).
* FeatureModifier (potentially add later, to allow encouraging certain features more)

#### TileBiome (PolyGridSubTile) Attributes


Need to be capturing things like:
* Elevation
* Wetness ??
* Humidity
* Temperature
* List of Features (Eg forests)
* Terrain (class) - calculated from the other attributes.

And likely others. 

These will be populated procedurarlly, utilising attributes from the parent and used to help the procedural generation of tiles in some cases.

### Procedural Generation

This will cover how we assign values from the various attributes to tiles on a globe. Some may need to be run sequentially, as results from some may affect others, eg humidity affected by water level. actually maybe it's mainly only elevation that needs the procedrual path? and then some features as well. but i feel like if we nail the elevation generation, other attributes can be calculated from that and the parent attributes potentially.

#### Elevation
Rules for elevation


When creating a region, we'll work at a low level grid tile for the sub-section polygrid, running the algorithim on that large interconnected grid. 
When building the world initially, we'll try a simpler approach i think as it may be too complex to do the full interconnected polygrid - assigning values at goldberg polyhedon tile first, then some other algorithms / clean up tasks run that smooth things out a bit and make it cohesive, paying attention to not look too 'hexagonal', which is a danger with this approach.

### Terrains

Build this into a table, with listed things that effect them eg temperature, water level, 

* Ocean

* Mountains
* Hills
* Plains
* Desert
* Marsh

#### Ocean

Then per section break down of key information about rules to decide on which biome is assigned.



### Features

Need to think about this. I think they basically run as procedural generators after everything else when creating a region. 

* Coast
* Lakes
* Rivers
* Forests
* Cities
* 


