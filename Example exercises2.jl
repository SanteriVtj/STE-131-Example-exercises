### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ bf822e2a-f5a1-49f6-9a3d-2a326e85facd
begin
	using JuMP
	using HiGHS
	using Random
	using Distributions
	using CairoMakie
	using DataFrames
	using PlutoUI
	using LaTeXStrings
	using ArchGDAL
	using GeoDataFrames
	using GeometryOps
	using GeoInterface
	using Distances
	using LinearAlgebra
	using Statistics
	using PairPlots
	using Statistics

	import GeometryOps as GO
	import GeoInterface as GI

	Random.seed!(123)
end

# ╔═╡ 300bf605-8503-4f0e-86c0-9616c7c863fe
TableOfContents()

# ╔═╡ 3aae6e9b-54c3-40ce-a411-65a60879c713
@bindname fixed_logistics_cost PlutoUI.Slider(
	0:10:500;
	default=200,
	show_value=true
)

# ╔═╡ e664545c-abbe-4c7a-b2b2-c14614ecaa12
md"""
The fixed logistics costs controls the price of delivering the gypsym to each field, or to each delivery point.
"""

# ╔═╡ c73bbb70-56fc-4f2a-9a0f-38666bcd1faa
@bindname spatial_correlation_range PlutoUI.Slider(
	100:100:5000;
	default=1500,
	show_value=true
)

# ╔═╡ d95deda5-a9d8-45fc-8ec7-4c8fdffb8638
md"""
The spatial correlation range controls how strongly fields at different distances are correlated. The correlation matrix is
```math
	R_{ij}=\exp\left(-\frac{d_{ij}}{\ell}\right)
```
where ``\ell`` corresponds to `spatial_correlation_range` and ``d_{ij}`` denotes the distance between fields ``i`` and ``j``. Therefore, the correlation between fields decreases exponentially with distance.
"""

# ╔═╡ 59e6d996-3a62-4b02-8774-be424307de15
@bindname ρₛ PlutoUI.Slider(
	-1:.01:1,
	default = .5,
	show_value=true
)

# ╔═╡ b6a06a94-b6fe-45d3-9d1d-3a7fbc8a90bc
@bindname ρₗ PlutoUI.Slider(
	-1:.01:1,
	default = .25,
	show_value=true
)

# ╔═╡ 3a979d09-3d6d-4398-85a2-c380f70a170d
begin
	function overlaps_bbox(geom, xmin, xmax, ymin, ymax)
		ext = GI.extent(geom)

		gxmin = ext.X[1]
		gxmax = ext.X[2]
		gymin = ext.Y[1]
		gymax = ext.Y[2]

		return (
			gxmax ≥ xmin &&
			gxmin ≤ xmax &&
			gymax ≥ ymin &&
			gymin ≤ ymax
		)
	end

	function min_geom_distance(geom, geometries)
		isempty(geometries) && return Inf

		minimum(
			GO.distance(geom, g)
			for g in geometries
		)
	end

	function spatial_normal(R; nugget = 1e-8)
		n = size(R, 1)

		L = cholesky(
			Symmetric(R + nugget * I)
		).L

		return L * randn(n)
	end

	function water_class(
		d_surface,
		d_ditch;
		threshold = 30.0,
	)
		if d_surface ≤ threshold
			return "Surface water"
		elseif d_ditch ≤ threshold
			return "Ditch"
		else
			return "Land"
		end
	end

	function draw_geometry!(
		ax,
		geom;
		color = (:yellow, 0.2),
		colormap = nothing,
		colorrange = nothing,
		strokecolor = :black,
		strokewidth = 1.0,
	)

		coords = GI.coordinates(geom)
		trait = GI.geomtrait(geom)

		function draw_polygon!(exterior)
			points = Point2f[
				Point2f(p[1], p[2])
				for p in exterior
			]

			if isnothing(colormap) || isnothing(colorrange)
				poly!(
					ax,
					points;
					color,
					strokecolor,
					strokewidth,
				)
			else
				poly!(
					ax,
					points;
					color,
					colormap,
					colorrange,
					strokecolor,
					strokewidth,
				)
			end
		end

		if trait isa GI.PolygonTrait

			draw_polygon!(coords[1])

		elseif trait isa GI.MultiPolygonTrait

			for polygon in coords
				draw_polygon!(polygon[1])
			end
		end

		return nothing
	end

	function add_geometry_information(df)
		out = copy(df)

		centroid_area = [
			GO.centroid_and_area(g)
			for g in out.geometry
		]

		out.x = [
			first(ca[1])
			for ca in centroid_area
		]

		out.y = [
			last(ca[1])
			for ca in centroid_area
		]

		out.area_ha = [
			abs(ca[2]) / 10_000
			for ca in centroid_area
		]

		return out
	end

	function arcdal_min_geom_distance(geom, geometries)
		isempty(geometries) && return Inf

		minimum(
			ArchGDAL.distance(geom, g)
			for g in geometries
		)
	end

	function adjacency_matrix(geometries; maximum_gap = 50.0)
		n = length(geometries)

		A = falses(n, n)

		for i in 1:(n - 1)
			for j in (i + 1):n

				d = ArchGDAL.distance(
					geometries[i],
					geometries[j],
				)

				if 0.0 ≤ d ≤ maximum_gap
					A[i, j] = true
					A[j, i] = true
				end
			end
		end

		return A
	end

	function phosphorus_EBI(
		P,
		slope,
		location,
		table,
	)
		pclass =
			P < 8 ? 1 :
			P ≤ 14 ? 2 :
			3

		sclass =
			slope < 1.5 ? 1 :
			slope ≤ 6 ? 2 :
			3

		lclass =
			location == "Land" ? 1 :
			location == "Ditch" ? 2 :
			3

		return table[
			sclass,
			lclass,
			pclass,
		]
	end

	function assemble_auction_data(
		fields;
		dist_surface,
		dist_ditch,
		water_type,
		P_status,
		slope,
		EBI,
		benefit,
		bid_ha,
		bid,
		neighbors,
	)
		out = copy(fields)

		out.dist_surface = dist_surface
		out.dist_ditch = dist_ditch
		out.water_type = water_type
		out.P_status = P_status
		out.slope = slope
		out.EBI = EBI
		out.benefit = benefit
		out.bid_ha = bid_ha
		out.bid = bid
		out.neighbors = neighbors

		return out
	end

	function allocation_summary(
		fields,
		selected,
		name,
	)
		chosen = selected .== 1

		return (
			method = name,
			n_fields = sum(chosen),
			area_ha = sum(fields.area_ha[chosen]),
			bid = sum(fields.bid[chosen]),
			benefit = sum(fields.benefit[chosen]),
			mean_EBI = mean(fields.EBI[chosen]),
			mean_P = mean(fields.P_status[chosen]),
			mean_water_distance =
				mean(fields.dist_surface[chosen])
		)
	end

	map = ArchGDAL.read(joinpath(@__DIR__, "L3341E.png"))
	gt = ArchGDAL.getgeotransform(map)

	x0 = gt[1]
	dx = gt[2]
	y0 = gt[4]
	dy = gt[6]
	
	nx = ArchGDAL.width(map)
	ny = ArchGDAL.height(map)
	
	xmin = x0
	xmax = x0 + nx * dx
	ymax = y0
	ymin = y0 + ny * dy
	
	(xmin, xmax, ymin, ymax)

	map_image = ArchGDAL.imread(map)

	# Downsample map
	step = 2
    map_small = map_image[1:step:end, 1:step:end]
    map_rotated = map_rotated = reverse(permutedims(map_small), dims = 2)

	fields_raw = GeoDataFrames.read(
	    joinpath(@__DIR__, "auction_fields.gpkg");
	    layer = "maatalousmaa",
	)

	inside = [
        overlaps_bbox(g, xmin, xmax, ymin, ymax)
        for g in fields_raw.geometry
    ]
    
    fields_map = fields_raw[inside, :]


	map_gt = ArchGDAL.getgeotransform(map)

	map_x0 = map_gt[1]
	map_dx = map_gt[2]
	map_y0 = map_gt[4]
	map_dy = map_gt[6]

	map_nx = ArchGDAL.width(map)
	map_ny = ArchGDAL.height(map)

	map_xmin = map_x0
	map_xmax = map_x0 + map_nx * map_dx
	map_ymax = map_y0
	map_ymin = map_y0 + map_ny * map_dy

	map_image_raw = ArchGDAL.imread(map)

	map_step = 2

	map_image_small =
		map_image_raw[1:map_step:end, 1:map_step:end]

	map_image_plot =
		reverse(permutedims(map_image_small), dims = 2)

	fields_inside_mask = [
		overlaps_bbox(
			g,
			map_xmin,
			map_xmax,
			map_ymin,
			map_ymax,
		)
		for g in fields_raw.geometry
	]

	fields_map_raw = fields_raw[fields_inside_mask, :]

	fields_geometry = add_geometry_information(fields_map_raw)

	minimum_field_area = 0.0

	fields_clean = filter(
		:area_ha => >=(minimum_field_area),
		fields_geometry,
	)

	fields_clean.id = 1:nrow(fields_clean)

	lakes_raw = GeoDataFrames.read(
		joinpath(@__DIR__, "auction_lakes.gpkg");
		layer = "jarvi",
	)

	rivers_raw = GeoDataFrames.read(
		joinpath(@__DIR__, "auction_rivers.gpkg");
		layer = "virtavesialue",
	)

	ditches_raw = GeoDataFrames.read(
		joinpath(@__DIR__, "auction_ditches.gpkg");
		layer = "virtavesikapea",
	)

	sea_raw = GeoDataFrames.read(
		joinpath(@__DIR__, "auction_sea.gpkg");
		layer = "meri",
	)

	lakes_map = lakes_raw[
		[
			overlaps_bbox(
				g,
				map_xmin,
				map_xmax,
				map_ymin,
				map_ymax,
			)
			for g in lakes_raw.geometry
		],
		:
	]

	rivers_map = rivers_raw[
		[
			overlaps_bbox(
				g,
				map_xmin,
				map_xmax,
				map_ymin,
				map_ymax,
			)
			for g in rivers_raw.geometry
		],
		:
	]

	ditches_map = ditches_raw[
		[
			overlaps_bbox(
				g,
				map_xmin,
				map_xmax,
				map_ymin,
				map_ymax,
			)
			for g in ditches_raw.geometry
		],
		:
	]

	sea_map = sea_raw[
		[
			overlaps_bbox(
				g,
				map_xmin,
				map_xmax,
				map_ymin,
				map_ymax,
			)
			for g in sea_raw.geometry
		],
		:
	]

	surface_water_geoms = vcat(
		collect(lakes_map.geometry),
		collect(rivers_map.geometry),
		collect(sea_map.geometry),
	)

	ditch_geoms = collect(ditches_map.geometry)

	field_dist_surface = [
		arcdal_min_geom_distance(
			g,
			surface_water_geoms,
		)
		for g in fields_clean.geometry
	]

	field_dist_ditch = [
		arcdal_min_geom_distance(
			g,
			ditch_geoms,
		)
		for g in fields_clean.geometry
	]

	field_water_type = [
		water_class(
			field_dist_surface[i],
			field_dist_ditch[i];
			threshold = 30.0,
		)
		for i in 1:nrow(fields_clean)
	]

	field_coordinates = Matrix(
		fields_clean[:, [:x, :y]]
	)

	field_distance_matrix = pairwise(
		Euclidean(),
		field_coordinates',
		dims = 2,
	)

	### Data generation ###
	field_spatial_R =
		exp.(
			-field_distance_matrix ./
			spatial_correlation_range
		)

	field_adjacency = adjacency_matrix(
		fields_clean.geometry;
		maximum_gap = 50.0,
	)

	field_edges = [
		(i, j)
		for i in 1:(nrow(fields_clean) - 1)
		for j in (i + 1):nrow(fields_clean)
		if field_adjacency[i, j]
	]

	field_neighbor_count =
		vec(sum(field_adjacency, dims = 2))

	latent_P =
		spatial_normal(field_spatial_R)

	latent_slope_independent =
		spatial_normal(field_spatial_R)

	latent_cost_independent =
		spatial_normal(field_spatial_R)

	latent_noise =
		randn(nrow(fields_clean))

	P_uniform =
		cdf.(Normal(), latent_P)

	P_distribution =
		Gamma(5.4, 2.2)

	field_P_status = round.(
		quantile.(P_distribution, P_uniform),
		digits = 1,
	)

	latent_slope =
		ρₛ .* latent_P .+
		sqrt(1 - ρₛ^2) .* latent_slope_independent

	field_slope = round.(
		clamp.(
			exp.(
				log(1.8) .+
				0.55 .* latent_slope
			),
			0.2,
			10.0,
		),
		digits = 2,
	)

	EBI_lowP = [
		 8  17  19
		14  30  34
		29  66  73
	]

	EBI_midP = [
		13  28  31
		18  42  46
		34  77  85
	]

	EBI_highP = [
		18  41   46
		24  55   61
		40  90  100
	]

	EBI_table = cat(
		EBI_lowP,
		EBI_midP,
		EBI_highP;
		dims = 3,
	)

	field_EBI = [
		phosphorus_EBI(
			field_P_status[i],
			field_slope[i],
			field_water_type[i],
			EBI_table,
		)
		for i in 1:nrow(fields_clean)
	]

	field_total_benefit =
		field_EBI .* fields_clean.area_ha

	latent_cost =
		ρₗ .* latent_P .+
		sqrt(1 - ρₗ^2) .* latent_cost_independent

	field_variable_bid_ha = round.(
		clamp.(
			200 .+
			12 .* latent_cost .+
			0.20 .* field_EBI .+
			6 .* latent_noise,
			180.0,
			290.0,
		),
		digits = 0,
	)

	field_total_bid = field_variable_bid_ha .* fields_clean.area_ha

	field_bid_ha =
		field_total_bid ./ fields_clean.area_ha

	fields_auction = assemble_auction_data(
		fields_clean;
		dist_surface = field_dist_surface,
		dist_ditch = field_dist_ditch,
		water_type = field_water_type,
		P_status = field_P_status,
		slope = field_slope,
		EBI = field_EBI,
		benefit = field_total_benefit,
		bid_ha = field_bid_ha,
		bid = field_total_bid,
		neighbors = field_neighbor_count,
	)
end;

# ╔═╡ ac4d4716-94f6-410d-a136-f8ffabbb63f0
begin
	max_budget = Int(ceil(sum(fields_auction.bid)))
	@bindname auction_budget PlutoUI.Slider(
		0:1000:max_budget; 
		default=50000, 
		show_value=true
	)
end

# ╔═╡ a1c85d82-1ac6-4981-97d9-c3afab3ee599
begin
    share_of_bids = round(auction_budget/sum(fields_auction.bid)*100; digits=1);
    repeat(
        "💰", 
        clamp(Int(round(auction_budget/round(max_budget/10))),0,10)
    )*repeat(
        " ■",
        10-clamp(Int(round(auction_budget/round(max_budget/10))),0,10)
    )
end

# ╔═╡ c1a566f0-9650-40aa-818e-707a141e77b6
md"""
The current budget of $(auction_budget)€ corresponds to $(share_of_bids)% out of the total bids.
"""

# ╔═╡ 8f5dc258-762b-4bfa-a673-9478016c5e4a
md"""
The parameter ``\rho_s`` controls the correlation of the slope's latent variable with the phosphorus state so that ``\text{Corr}(z^S,z^P)=\rho_s`` (see the "Details of data generating process" for more information). The resulting correlation between slopes and phosphorus level is then $(round(cor(Matrix(fields_auction[!, [:P_status, :slope]]))[1,2], digits=3)). As the correlation governs the relationsip between the hidden states, it is not to be expected that the correlation between the slope and phosphorus would exactly match the ``\rho_s``.
"""

# ╔═╡ 58f9c253-314a-4154-8e24-7cc7637809da
md"""
The parameter ``\rho_l`` controls the correlation between the latent state of costs and phosphorus similarly so that ``\text{Corr}(z^C,z^P)=\rho_l``. The cost shock directly enters the bid calculation (see the "Details of data generating process for more information).
"""

# ╔═╡ c007daea-26d6-4f23-8aeb-1a0db6af1091
begin
	fig_fields = Figure(size = (900, 900))

	ax_fields = Axis(
		fig_fields[1, 1],
		xlabel = "Easting (m)",
		ylabel = "Northing (m)",
		aspect = DataAspect(),
		title = "Agricultural fields",
	)

	image!(
		ax_fields,
		(map_xmin, map_xmax),
		(map_ymin, map_ymax),
		map_image_plot,
	)

	for geom in fields_clean.geometry
		draw_geometry!(
			ax_fields,
			geom;
			color = (:yellow, 0.18),
			strokecolor = :red,
			strokewidth = 0.8,
		)
	end

	xlims!(ax_fields, map_xmin, map_xmax)
	ylims!(ax_fields, map_ymin, map_ymax)

	fig_fields

	###########################################
	#### Spatial distribution of variables ####
	###########################################
	fig = Figure(size = (1200, 1100))

	### Layout ###
	grid_P     = fig[1, 1] = GridLayout()
	grid_slope = fig[1, 2] = GridLayout()
	grid_bid   = fig[2, 1] = GridLayout()
	grid_EBI   = fig[2, 2] = GridLayout()


	### P plot ###
	ax_P = Axis(
		grid_P[1, 1],
		xlabel = "Easting (m)",
		ylabel = "Northing (m)",
		aspect = DataAspect(),
		title = "Simulated soil P-status",
	)

	image!(
		ax_P,
		(map_xmin, map_xmax),
		(map_ymin, map_ymax),
		map_image_plot,
	)

	P_range = extrema(fields_auction.P_status)
	P_colormap = :viridis

	for i in 1:nrow(fields_auction)
		draw_geometry!(
			ax_P,
			fields_auction.geometry[i];
			color = fields_auction.P_status[i],
			colormap = (P_colormap, 0.4),
			colorrange = P_range,
			strokecolor = :black,
			strokewidth = 0.5,
		)
	end

	Colorbar(
		grid_P[1, 2],
		limits = P_range,
		colormap = P_colormap,
		label = "P-status (mg/L)",
	)

	xlims!(ax_P, map_xmin, map_xmax)
	ylims!(ax_P, map_ymin, map_ymax)


	### Slope plot ###
	ax_slope = Axis(
		grid_slope[1, 1],
		xlabel = "Easting (m)",
		ylabel = "Northing (m)",
		aspect = DataAspect(),
		title = "Simulated slope",
	)

	image!(
		ax_slope,
		(map_xmin, map_xmax),
		(map_ymin, map_ymax),
		map_image_plot,
	)

	slope_range = extrema(fields_auction.slope)
	slope_colormap = :viridis

	for i in 1:nrow(fields_auction)
		draw_geometry!(
			ax_slope,
			fields_auction.geometry[i];
			color = fields_auction.slope[i],
			colormap = (slope_colormap, 0.4),
			colorrange = slope_range,
			strokecolor = :black,
			strokewidth = 0.5,
		)
	end

	Colorbar(
		grid_slope[1, 2],
		limits = slope_range,
		colormap = slope_colormap,
		label = "Slope",
	)

	xlims!(ax_slope, map_xmin, map_xmax)
	ylims!(ax_slope, map_ymin, map_ymax)


	### Bid plot ###
	ax_bid = Axis(
		grid_bid[1, 1],
		xlabel = "Easting (m)",
		ylabel = "Northing (m)",
		aspect = DataAspect(),
		title = "Simulated bids",
	)

	image!(
		ax_bid,
		(map_xmin, map_xmax),
		(map_ymin, map_ymax),
		map_image_plot,
	)

	bid_range = extrema(fields_auction.bid)
	bid_colormap = :viridis

	for i in 1:nrow(fields_auction)
		draw_geometry!(
			ax_bid,
			fields_auction.geometry[i];
			color = fields_auction.bid[i],
			colormap = (bid_colormap, 0.4),
			colorrange = bid_range,
			strokecolor = :black,
			strokewidth = 0.5,
		)
	end

	Colorbar(
		grid_bid[1, 2],
		limits = bid_range,
		colormap = bid_colormap,
		label = "Bid",
	)

	xlims!(ax_bid, map_xmin, map_xmax)
	ylims!(ax_bid, map_ymin, map_ymax)


	### EBI plot ###
	ax_EBI = Axis(
		grid_EBI[1, 1],
		xlabel = "Easting (m)",
		ylabel = "Northing (m)",
		aspect = DataAspect(),
		title = "Simulated environmental benefit index",
	)

	image!(
		ax_EBI,
		(map_xmin, map_xmax),
		(map_ymin, map_ymax),
		map_image_plot,
	)

	EBI_range = extrema(fields_auction.EBI)
	EBI_colormap = :viridis

	for i in 1:nrow(fields_auction)
		draw_geometry!(
			ax_EBI,
			fields_auction.geometry[i];
			color = fields_auction.EBI[i],
			colormap = (EBI_colormap, 0.4),
			colorrange = EBI_range,
			strokecolor = :black,
			strokewidth = 0.5,
		)
	end

	Colorbar(
		grid_EBI[1, 2],
		limits = EBI_range,
		colormap = EBI_colormap,
		label = "EBI",
	)

	xlims!(ax_EBI, map_xmin, map_xmax)
	ylims!(ax_EBI, map_ymin, map_ymax)


	rowgap!(fig.layout, 15)
	colgap!(fig.layout, 30)

	fig

	### Gamma plot ###
	dist = Gamma(5.4,2.2)

	x = range(0, 35; length = 500)
	y = pdf.(dist, x)

	gamma_fig = Figure(size = (700, 450))

	gamma_ax = Axis(
		gamma_fig[1, 1],
		xlabel = "P-status",
		ylabel = "Density",
		title = L"\Gamma(5.4,\,2.2)",
		xgridvisible = false,
		ygridvisible = false,
	)

	lines!(
		gamma_ax,
		x,
		y;
		linewidth = 3,
		color=:orangered3
	)

	band!(
		gamma_ax,
		x,
		zeros(length(x)),
		y;
		alpha = 0.15,
		color=:orangered3
	)

	vlines!(
		gamma_ax,
		[mean(dist)];
		linestyle = :dash,
		linewidth = 2,
		color=:black,
		label = "Mean = $(round(mean(dist), digits = 2))",
	)

	axislegend(gamma_ax)

	xlims!(gamma_ax, 0, 35)
	ylims!(gamma_ax, 0, nothing)

	hidespines!(gamma_ax, :t, :r)

	gamma_fig

	### Adjacency structure ###
	schem_x = [0.0, 1.5, 3.0, 4.5, 1.5, 3.0]
	schem_y = [0.0, 0.0, 0.0, 0.0, 1.4, 1.4]

	schem_edges = [
		(1, 2),
		(2, 3),
		(3, 4),
		(2, 5),
		(3, 6),
	]

	schem_selected_idx = 1:6

	schem_logistics_idx = [2, 3]

	schem_regular_idx = setdiff(collect(schem_selected_idx), schem_logistics_idx)

	schem_fig = Figure(size = (800, 400))

	schem_ax = Axis(
		schem_fig[1, 1],
		title = "Adjacency-based logistics",
		aspect = DataAspect(),
		xgridvisible = false,
		ygridvisible = false,
		xticksvisible = false,
		yticksvisible = false,
		xticklabelsvisible = false,
		yticklabelsvisible = false,
	)
	
	schem_first_link = true

	for (schem_i, schem_j) in schem_edges
		lines!(
			schem_ax,
			[schem_x[schem_i], schem_x[schem_j]],
			[schem_y[schem_i], schem_y[schem_j]];
			color = :red,
			linewidth = 2,
			label = schem_first_link ? "Adjacency link" : nothing,
		)

		schem_first_link = false
	end
	
	scatter!(
		schem_ax,
		schem_x[schem_regular_idx],
		schem_y[schem_regular_idx];
		marker = :circle,
		markersize = 42,
		color = (:green, 0.55),
		strokecolor = :black,
		strokewidth = 1.5,
		label = "Selected field",
	)

	scatter!(
		schem_ax,
		schem_x[schem_logistics_idx],
		schem_y[schem_logistics_idx];
		marker = :circle,
		markersize = 42,
		color = (:blue, 0.75),
		strokecolor = :black,
		strokewidth = 1.5,
		label = "Logistics point",
	)

	for schem_i in 1:length(schem_x)
		text!(
			schem_ax,
			schem_x[schem_i],
			schem_y[schem_i];
			text = string(schem_i),
			align = (:center, :center),
			fontsize = 16,
			color = :black,
		)
	end

	schem_field_legend = MarkerElement(
	    marker = :circle,
	    color = (:green, 0.55),
	    strokecolor = :black,
	    markersize = 18,
	)
	
	schem_logistics_legend = MarkerElement(
	    marker = :circle,
	    color = (:blue, 0.75),
	    strokecolor = :black,
	    markersize = 18,
	)
	
	schem_adjacency_legend = LineElement(
	    color = :red,
	    linewidth = 2,
	)
	
	axislegend(
	    schem_ax,
	    [
	        schem_adjacency_legend,
	        schem_field_legend,
	        schem_logistics_legend,
	    ],
	    [
	        "Adjacency link",
	        "Selected field",
	        "Logistics point",
	    ];
	    position = :rt,
	)

	xlims!(schem_ax, -0.8, 5.3)
	ylims!(schem_ax, -0.8, 2.6)

	hidespines!(schem_ax)

	schem_fig
end;

# ╔═╡ b2a3713d-e71f-405a-b84d-83b66a703ffc
md"""
# Example exercise: choosing an optimal allocation from environmental auction with logistics costs

## Exercise 2.

All of the spatial data used in this notebook are downloaded from NLS [^1], while other variables, such as phosphorus, slope, and bids, are simulated, drawing inspiration from the _Iho et al._ [^2] paper on agri-environmental auctions for phosphorus load reduction. While the paper focuses more on the auction design and bidders' strategies, at this point we are only interested in optimizing the selection of bids. The data-generating process also uses some statistics published in the article.

Imagine that you are an auctioneer tasked with implementing an auction for treating agricultural land with gypsum. Each eligible farmer in the selected treatment area near Salo is given the opportunity to submit a bid indicating the compensation they would require to spread gypsum on their fields in order to reduce phosphorus loads to nearby waters. However, you are still responsible for delivering the gypsum to the treatment locations. For each delivery, you need to pay the amount specified by `fixed_logistics_cost`. This fixed cost can, however, be shared between multiple fields if those fields are at most 50 m apart from each other.

$(fig_fields)

**Figure 1.** Map of the auction site, where all fields are highlighted.

Figure 1 shows that many of the fields are adjacent to each other. Depending on the logistics costs, selecting the fields to include in the treatment strategically could therefore substantially affect the efficiency of the solution.

The objective is to maximize the acquired environmental benefit index (EBI) for a given budget. The EBI is constructed as a function of the phosphorus content of the field, the slope of the field, and its location relative to water. It is computed according to [^2] using the following values:

**Environmental Benefit Index (EBI)**

| P-status | **Slope <1.5%** |  |  | **Slope 1.5–6%** |  |  | **Slope >6%** |  |  |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
|  | Land | Ditch | Surface water | Land | Ditch | Surface water | Land | Ditch | Surface water |
| **< 8** | 8 | 17 | 19 | 14 | 30 | 34 | 29 | 66 | 73 |
| **8–14** | 13 | 28 | 31 | 18 | 42 | 46 | 34 | 77 | 85 |
| **> 14** | 18 | 41 | 46 | 24 | 55 | 61 | 40 | 90 | 100 |

For example, if a field has P-status ``8\leq P\leq14``, slope ``1.5\leq S\leq6``, and is classified as bordering a ditch, the field has ``EBI=42``. The total benefit of field ``i`` is then the EBI multiplied by the area of the field, ``\text{benefit}_i=A_iEBI_i``.

$(fig)

**Figure 2.** Map of the auction site showing phosphorus values, slope values, and the computed EBI.

Figure 2 shows how the relevant variables and the resulting EBI are distributed spatially across the auction site. You can read more about how the simulated data are constructed below. You can also modify some of the parameters used in the data-generating process and the optimization.

The dataset available to you is as follows:

$(first(fields_auction,5))

[^1]: National Land Survey of Finland, [MapSite](https://asiointi.maanmittauslaitos.fi/karttapaikka/?lang=en)
[^2]: Iho, A., Lankoski, J., Ollikainen, M., Puustinen, M. and Lehtimäki, J. (2014), Agri-environmental auctions for phosphorus load reduction: experiences from a Finnish pilot†. Aust J Agric Resour Econ, 58: 205-222. [https://doi.org/10.1111/1467-8489.12049](https://doi.org/10.1111/1467-8489.12049)
"""

# ╔═╡ cfc8c30d-7dd2-4899-9fe7-678e6476d1c0
PlutoUI.details(
"Details of data-generating process",
md"""
The underlying idea of the data-generating process is to create some interesting spatial correlation between the variables. This should also affect how the bids are selected, since the EBI values should be somewhat spatially clustered. As a result, logistics costs can potentially be reduced by selecting nearby fields that have relatively high EBI values.

Let ``s_i`` and ``s_j`` denote the centroids of fields ``i`` and ``j``. The distance between the two fields is computed as
```math
	d_{ij}=\|s_i-s_j\|.
```
This distance is used to construct the covariance matrix
```math
	R_{ij}=\exp\left(-\frac{d_{ij}}{\ell}\right),
```
where ``\ell`` controls the strength and range of the spatial correlation. In the code, this parameter is denoted by `spatial_correlation_range`.

The covariance matrix is used to generate a Gaussian latent variable
```math
	z\sim\mathcal{N}(0,R),
```
so that nearby fields have, on average, more similar values. Each variable denoted by ``z`` below follows this distribution.

This latent state is used to generate the phosphorus values using a Gaussian copula as follows:
```math
	P_i=F^{-1}_{\Gamma(5.4,2.2)}(\Phi(z^P_i)),
```
so that the marginal distribution of the phosphorus values approximately follows ``P_i\sim\Gamma(5.4,2.2)``. This particular parametrization is chosen to correspond to the values observed in Uusimaa in [^2]. Figure 3 shows the marginal distribution of phosphorus.

$(gamma_fig)

**Figure 3.** Density of the ``\Gamma(5.4,2.2)`` distribution.

The slope is generated as a linear combination of the phosphorus latent variable and another draw from the ``z`` distribution:
```math
	z_i^S=0.25z_i^P+\sqrt{1-0.25^2}z_i^{S},
```
which gives it a correlation of 0.25 [^3] with the phosphorus latent variable. The corresponding slope values are then computed as
```math
	S_i=\text{min}\{\text{max}\{\exp(\log(1.8)+0.55z_i^S),0.2\},10\}.
```
These values are then used to compute the EBI as described above.

The final per-hectare bids are computed by correlating one additional latent state with the phosphorus latent state:
```math
	z_i^C=0.20z_i^P+\sqrt{1-0.20^2}z_i^{C},
```
and setting the per-hectare bid as
```math
	b_i^{ha}=\text{min}\{\text{max}\{
		200+12z_i^C+0.2EBI_i+6\epsilon_i,
		0.2\},
	10\},
```
where ``z_i^C`` represents a spatially correlated cost component and ``\epsilon_i\sim\mathcal{N}(0,1)`` is an independent random shock.


[^3] The correlation behaves as described because, for two independent standard normal random variables ``X`` and ``Y``, if ``Z=\rho X+\sqrt{1-\rho^2}Y``, then
```math
	\text{Var}(Z)
	=
	\rho^2\text{Var}(X)
	+
	(1-\rho^2)\text{Var}(Y)
	=
	1,
```
and
```math
	\text{Cov}(X,Z)
	=
	\text{Cov}\left(X,\rho X+\sqrt{1-\rho^2}Y\right)
	=
	\rho\text{Var}(X)
	+
	\sqrt{1-\rho^2}\text{Cov}(X,Y)
	=
	\rho.
```
This means that the constructed random variable ``Z`` is also marginally standard normal, while satisfying ``\text{Corr}(X,Z)=\rho`` and ``\text{Corr}(Y,Z)=\sqrt{1-\rho^2}``.
"""
)

# ╔═╡ 48edb189-23ad-411f-be62-05d5c669fd9f
md"""
### a)

> Plot the variables `P_status`, `slope`, `bid_ha`, `EBI`, and `dist_surface` from `fields_auction` as a pairs plot. These correspond to the phosphorus value in mg/L, slope in %, bid per hectare in €, environmental benefit index, and the minimum distance from each field to surface water.

### b)

> Implement an optimization function similar to the one in Exercise 1 that maximizes the total environmental benefit while satisfying the budget constraint. Be sure to include the logistics costs!

### c)

> Implement an optimization function that still maximizes the total environmental benefit, but now make use of the fact that neighboring fields can share a delivery point. You can model this by paying the logistics cost only once for each selected delivery point.
>
> For example, consider six fields with the adjacency structure shown in the plot below. The logistics points can be chosen as illustrated. Note that the optimal choice is not necessarily unique. For example, if fields 5 and 6 were absent, all of the following logistics-point choices would be equally good: `[(1,3), (2,4), (2,3)]`.
>
> Therefore, you cannot simply subtract a constant multiple of the neighborhood size from the fixed logistics cost. Instead, you need to introduce an auxiliary variable to represent the choice of logistics points.

$(schem_fig)

### d)

> Solve the optimization problem from part **c)** for multiple values of the budget and plot the value function ``V(B)``, i.e. the achieved environmental benefit as a function of the budget. What does the value function look like? Explain what you observe.
>
> Also plot an approximation of the derivative of the value function. At what budget level does an increase in the budget generate the largest increase in environmental benefit?
>
> Finally, plot the €/EBI ratio for each solution to examine how many euros must be spent to obtain one additional environmental benefit index point. Explain what you observe.

## Solutions

### a)

The plot below shows the distribution of each variable and the pairwise relationships between the variables using the `PairPlots.jl` package. Correlations between some variables, such as `slope` and `P_status`, can be clearly observed.

"""

# ╔═╡ ec2dd1c0-6613-4cbd-97fb-5c89fc95186d
pairplot(
	fields_auction[!,[:P_status, :slope, :bid_ha, :EBI, :dist_surface]] => (
		PairPlots.HexBin(colormap=Makie.cgrad([:transparent, "#333"])),
        PairPlots.Contour(linewidth=1.5),
        PairPlots.MarginHist(color=Makie.RGBA(0.4,0.4,0.4,0.15)),
        PairPlots.MarginStepHist(color=Makie.RGBA(0.4,0.4,0.4,0.8)),
        PairPlots.MarginDensity(
            color=:black,
            linewidth=1.5f0,
        ),
        PairPlots.MarginQuantileText(color=:black, font=:regular),
        PairPlots.MarginQuantileLines(),
		PairPlots.PearsonCorrelation()
	)
)

# ╔═╡ bd6b1a6d-c7bf-4c67-b962-aff24fe66eaf
md"""
### b)
"""

# ╔═╡ e8df9819-1a23-488d-b41b-3ffc8ca537d6
Markdown.parse(
"""
Formulation of the optimization problems is almost identical to the one in exercise 1. The only difference is that the fixed logistics cost needs to be taken into account when computing the budget constraint. This can be done as:

```math
\\begin{aligned}
	\\max_{x}\\quad & \\text{EBI}^\\top x \\\\
	\\text{s.t.}\\\\
	& \\text{bids}^\\top x + FC\\sum_{i=1}^Nx_i\\leq B \\\\
	& x_i\\in\\{0,1\\},\\quad \\forall i\\in\\{1,2,\\ldots,N\\}.
\\end{aligned}
```
where ``FC`` denotes the fixed cost form the logistics. The implementation is also very close to the one already seen.
"""
)

# ╔═╡ 432f5b73-f506-4ac8-af8e-0a30de194c94
begin
	function solve_basic_auction(fields, B, fixed_cost)
		n = nrow(fields)

		model = Model(HiGHS.Optimizer)

		# x[i] = 1 if field i is accepted
		@variable(model, x[1:n], Bin)

		# Maximize environmental benefit
		@objective(
			model,
			Max,
			sum(
				fields.benefit[i] * x[i]
				for i in 1:n
			)
		)

		# The sum of bids that are accepted need to satisfy budget constraint
		@constraint(
			model,
			sum(fields.bid[i] * x[i] for i in 1:n)
				+fixed_cost * sum(x[i] for i in 1:n)≤ B
		)

		optimize!(model)

		# Check result feasibility
	    if !is_solved_and_feasible(model)
	        return nothing
	    end

		# Return values
		x_value = round.(Int, value.(x))

		landowner_payments = sum(
	        fields.bid[i] * x_value[i]
	        for i in 1:n
	    )
	
	    logistics_cost =
	        fixed_cost * sum(x_value)
	
	    total_cost =
	        landowner_payments + logistics_cost

		return (
			x = x_value,
			objective = objective_value(model),
			total_cost = landowner_payments+logistics_cost
		)
	end
end

# ╔═╡ 61a1ab2f-5ad8-412e-a6c6-496b9f37f284
begin
	basic_solution =
		solve_basic_auction(
			fields_auction,
			auction_budget,
			fixed_logistics_cost
		)

	basic_selected =
		basic_solution.x
end;

# ╔═╡ 6fe45813-cfd7-4a7a-b052-6f98980d2298
md"""
### c)
"""

# ╔═╡ b44bd516-7e49-4e7b-a6e8-b91166799158
Markdown.parse(
"""
To account for the possible benefits from choosing the delivery points we need a new variable. The task for that variable is to denote those fields that are chosen as delivery points and it will also be another binary variable. The constraints it needs to satisfy are that ``z_i\\leq x_i`` so node cannot be a delivery point if it is not chosen from the auction and ``x_i\\leq z_i+\\sum_{i\\in NB}z_i``, where the set ``NB`` contains all x_i's neighbors.

```math
\\begin{aligned}
	\\max_{x}\\quad & \\text{EBI}^\\top x \\\\
	\\text{s.t.} \\\\
	& \\text{bids}^\\top x + FC\\sum_{i=1}^Nz_i\\leq B \\\\
	& x_i\\leq z_i+\\sum_{i\\in NB}z_i,\\quad\\forall i\\in\\{1,2,\\ldots,N\\} \\\\
	& z_i\\leq x_i,\\quad\\forall i\\in\\{1,2,\\ldots,N\\} \\\\
	& x_i\\in\\{0,1\\},\\quad \\forall i\\in\\{1,2,\\ldots,N\\} \\\\
	& z_i\\in\\{0,1\\},\\quad \\forall i\\in\\{1,2,\\ldots,N\\}
\\end{aligned}
```
The implementation is shown below.
"""
)

# ╔═╡ ab206d1d-6702-4f2a-b11a-f17066bc6cb1
function solve_spatial_auction(
    fields,
    adjacency,
    budget,
    fixed_cost
)
    n = nrow(fields)

    model = Model(HiGHS.Optimizer)

    # x[i] = 1 if field i is accepted
    @variable(model, x[1:n], Bin)

    # z[i] = 1 if field i is used as a logistics point
    @variable(model, z[1:n], Bin)

    # A logistics point can only be located at a field already selected
    @constraint(
        model,
        [i = 1:n],
        z[i] ≤ x[i]
    )

    # Every selected field must be served
    for i in 1:n
        neighbors = [
            j
            for j in 1:n
            if adjacency[i, j]
        ]

        @constraint(
            model,
            x[i] ≤ z[i] + sum(z[j] for j in neighbors)
        )
    end

    # The auctioneer faces costs from bids and logistics costs
    @constraint(
        model,
        sum(fields.bid[i] * x[i] for i in 1:n)
        +
        fixed_cost * sum(z[i] for i in 1:n) ≤ budget
    )

    # Maximize environmental benefit
    @objective(
        model,
        Max,
        sum(
            fields.benefit[i] * x[i]
            for i in 1:n
        )
    )

    optimize!(model)

    if !is_solved_and_feasible(model)
        return nothing
    end

    x_value = round.(Int, value.(x))
    z_value = round.(Int, value.(z))

    landowner_payments = sum(
        fields.bid[i] * x_value[i]
        for i in 1:n
    )

    logistics_cost =
        fixed_cost * sum(z_value)

    total_cost =
        landowner_payments + logistics_cost

    total_benefit = sum(
        fields.benefit[i] * x_value[i]
        for i in 1:n
    )

    return (
        x = x_value,
        z = z_value,
        objective = total_benefit,
        landowner_payments = landowner_payments,
        logistics_cost = logistics_cost,
        total_cost = total_cost,
        n_selected = sum(x_value),
        n_logistics_points = sum(z_value),
    )
end

# ╔═╡ 559b59e7-b6d8-4ae3-8957-cc98c55dc372
begin
	spatial_solution = solve_spatial_auction(
	    fields_auction,
	    field_adjacency,
	    auction_budget,
	    fixed_logistics_cost,
	)

	spatial_selected =
		spatial_solution.x
end;

# ╔═╡ 5a9b18f7-12ee-49b6-88e8-8fa1e93bef8f
begin
	allocation_results = DataFrame([
		allocation_summary(
			fields_auction,
			basic_selected,
			"Basic auction",
		),

		allocation_summary(
			fields_auction,
			spatial_selected,
			"Spatial bundling",
		),
	])

	allocation_results.EBI_per_1000_euro =
		1000 .* allocation_results.benefit ./
		allocation_results.bid

	allocation_results.total_cost = [
		basic_solution.total_cost, 
		spatial_solution.total_cost
	]

	allocation_results
end

# ╔═╡ ef41ea3f-c771-49b0-aabf-99dc00196302
begin
	fig_basic = Figure(size = (1200, 600))

	ax_basic = Axis(
		fig_basic[1, 1],
		xlabel = "Easting (m)",
		ylabel = "Northing (m)",
		aspect = DataAspect(),
		title = "Basic phosphorus auction",
	)

	image!(
		ax_basic,
		(map_xmin, map_xmax),
		(map_ymin, map_ymax),
		map_image_plot,
	)

	for i in 1:nrow(fields_auction)

		field_color =
			basic_selected[i] == 1 ?
			(:green, 0.5) :
			(:red, 0.2)

		draw_geometry!(
			ax_basic,
			fields_auction.geometry[i];
			color = field_color,
			strokecolor = :black,
			strokewidth = 0.7,
		)
	end

	chosen_element = PolyElement(
	    color = (:green, 0.5),
	    strokecolor = :black,
	    strokewidth = 0.7,
	)
	
	not_chosen_element = PolyElement(
	    color = (:red, 0.2),
	    strokecolor = :black,
	    strokewidth = 0.7,
	)
	
	axislegend(
	    ax_basic,
	    [chosen_element, not_chosen_element],
	    ["Chosen field", "Not chosen field"];
	    position = :lb,
	)

	xlims!(ax_basic, map_xmin, map_xmax)
	ylims!(ax_basic, map_ymin, map_ymax)

	ax_spat = Axis(
		fig_basic[1, 2],
		xlabel = "Easting (m)",
		ylabel = "Northing (m)",
		aspect = DataAspect(),
		title = "Spatial phosphorus auction",
	)

	image!(
		ax_spat,
		(map_xmin, map_xmax),
		(map_ymin, map_ymax),
		map_image_plot,
	)

	for i in 1:nrow(fields_auction)

		field_color =
			spatial_selected[i] == 1 ?
			(:green, 0.5) :
			(:red, 0.2)

		draw_geometry!(
			ax_spat,
			fields_auction.geometry[i];
			color = field_color,
			strokecolor = :black,
			strokewidth = 0.7,
		)
	end

	first_adj = true
	for (i, j) in field_edges

		if spatial_selected[i] == 1 &&
		   spatial_selected[j] == 1

			lines!(
				ax_spat,
				[
					fields_auction.x[i],
					fields_auction.x[j],
				],
				[
					fields_auction.y[i],
					fields_auction.y[j],
				],
				color = :red,
				linewidth = 2,
				label=first_adj ? "Adjacency link" : nothing
			)
			first_adj = false
		end
	end

	logistics_idx = findall(spatial_solution.z .== 1)

	scatter!(
	    ax_spat,
	    fields_auction.x[logistics_idx],
	    fields_auction.y[logistics_idx];
	    marker = :star5,
	    markersize = 18,
	    color = :blue,
	    strokecolor = :white,
	    strokewidth = 1.5,
	    label = "Logistics point",
	)

	axislegend(position=:lb)

	xlims!(ax_spat, map_xmin, map_xmax)
	ylims!(ax_spat, map_ymin, map_ymax)

	fig_basic
end

# ╔═╡ 30f18612-dcd2-455e-9df0-e5aaa95d4852
md"""
### d)

Let's start by computing the optimal allocations for budgets ranging from €10,000 to €75,000 in increments of €1,000.
"""

# ╔═╡ 829f8e9e-aa43-48cb-bb15-9150e0a4ff69
begin
	Bs = 1000:1000:75000
	V = Float64[]
	C = Float64[]
	for b in Bs
		result = solve_spatial_auction(
			fields_auction,
		    field_adjacency,
		    b,
		    fixed_logistics_cost,
		)
		push!(V, result.objective)
		push!(C, result.total_cost)
	end
end

# ╔═╡ 926b411d-40bd-424f-8074-40bf4fbd0985
md"""
Plotting these results yields the following figures. Initially, I expected the value function to be more step-like because the allocation decisions are binary. However, the function appears relatively smooth. This is probably because there are sufficiently many fields to choose from, allowing small changes in the budget to produce correspondingly small changes in the optimal allocation and its environmental benefit. The jaggedness can still be seen in the differences.
"""


# ╔═╡ 2cf97702-5204-4605-9da5-3e130be4dfff
begin
	V_fig = Figure(size=(1200,600))

	ax_v = Axis(
		V_fig[1,1],
		xlabel=L"B",
		ylabel=L"V(B)",
		title="Value function"
	)
	lines!(ax_v, Bs,V)

	ax_dv = Axis(
		V_fig[1,2],
		xlabel=L"B",
		ylabel=L"\frac{V(B+\Delta B)-V(B)}{\Delta B}",
		title="Discrete approximation of value function derivative"
	)
	lines!(ax_dv, Bs[1:end-1], diff(V) ./ 1000)

	V_fig
end

# ╔═╡ 4ea83ba8-85d9-4962-b20a-176593649e0d
begin
	index_of_highest_benefit = argmax(diff(V)) 
	highest_benefit = Bs[index_of_highest_benefit]
end;

# ╔═╡ 3b318522-990a-44d2-979b-29457e28df0b
md"""
The largest marginal increase in environmental benefit occurs when increasing the budget from $(Bs[index_of_highest_benefit])€ to $(Bs[index_of_highest_benefit+1])€, yielding an additional $(round(diff(V)[index_of_highest_benefit], digits=2)) environmental benefit index points.

The €/EBI ratio shown below initially decreases, but then begins to increase steadily. When the budget is very tight, the auctioneer cannot necessarily select the most cost-effective combination of bids because some bids are simply too large to fit within the available budget. In addition, even when several bids are feasible, the limited budget may prevent the solution from fully exploiting the benefits of sharing logistics points across neighboring fields.

As the budget increases, these constraints become less important and the allocation can make better use of spatial clustering and shared logistics costs. Beyond this point, however, further increases in the budget require selecting progressively less cost-effective fields with lower environmental benefits relative to their costs. As a result, the €/EBI ratio begins to rise steadily.
"""


# ╔═╡ b073829d-3b5f-49db-9ede-f3c4cf7c9b86
begin
	efficiecy_fig = Figure()

	ax_ef = Axis(
		efficiecy_fig[1,1],
		xlabel=L"B",
		ylabel=L"€/EBI",
		title="Cost of additional environmental benefit index point"
	)
	lines!(
		ax_ef,
		Bs,
		C./V
	)

	efficiecy_fig
end

# ╔═╡ 6af65f24-fdff-47cf-9d25-8736c4e43a9d
md"""
We can also take a closer look at how the selected fields differ from those that were not selected. The following plot compares the two groups in terms of the number of neighboring fields, phosphorus content, distance to surface water, per-hectare bid, and EBI.
"""

# ╔═╡ 1f7cb1bb-a5e4-486e-9e9f-ad7d4757e34e
begin
	property_fig = Figure(size=(1000,1000))

	nbh_ax = Axis(
		property_fig[1,1],
		ylabel=L"N",
		xlabel="Number of neighbors"
	)

	hist!(
		nbh_ax,
		fields_auction[spatial_selected.==1,:neighbors],
		label="Selected",
		bins=7,
		color=(:green, 0.3)
	)
	
	hist!(
		nbh_ax,
		fields_auction[spatial_selected.==0,:neighbors],
		label="Not selected",
		bins=7,
		color=(:red, 0.3)
	)

	axislegend(position=:rt)

	ec_ax = Axis(
		property_fig[1,2],
		xlabel="Bid €/ha",
		ylabel="EBI"
	)

	scatter!(
		ec_ax,
		fields_auction[spatial_selected.==0,:bid_ha],
		fields_auction[spatial_selected.==0,:EBI],
		color=(:red, 0.75),
		label="Not selected",
	)
	scatter!(
		ec_ax,
		fields_auction[spatial_selected.==1,:bid_ha],
		fields_auction[spatial_selected.==1,:EBI],
		color=(:green, 0.75),
		label="Selected",
	)
	
	axislegend(position=:lt)

	P_ax = Axis(
		property_fig[2,1],
		ylabel=L"N",
		xlabel="Phosphorus mg/L"
	)

	hist!(
		P_ax,
		fields_auction[spatial_selected.==1,:P_status],
		label="Selected",
		bins=7,
		color=(:green, 0.3)
	)
	
	hist!(
		P_ax,
		fields_auction[spatial_selected.==0,:P_status],
		label="Not selected",
		bins=7,
		color=(:red, 0.3)
	)

	axislegend(position=:rt)

	dist_ax = Axis(
		property_fig[2,2],
		ylabel=L"N",
		xlabel="Distance to water m"
	)

	hist!(
		dist_ax,
		fields_auction[spatial_selected.==1,:dist_surface],
		label="Selected",
		bins=7,
		color=(:green, 0.3)
	)
	
	hist!(
		dist_ax,
		fields_auction[spatial_selected.==0,:dist_surface],
		label="Not selected",
		bins=7,
		color=(:red, 0.3)
	)

	axislegend(position=:rt)

	property_fig
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ArchGDAL = "c9ce4bd3-c3d5-55b8-8973-c0e20141b8c3"
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
GeoDataFrames = "62cb38b5-d8d2-4862-a48e-6a340996859f"
GeoInterface = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
GeometryOps = "3251bfac-6a57-4b6d-aa61-ac1fef2975ab"
HiGHS = "87dc4568-4c63-4d18-b0c0-bb2238e4078b"
JuMP = "4076af6c-e467-56ae-b986-b466b2749572"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PairPlots = "43a3c2be-4208-490b-832a-a21dcd55d7da"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
ArchGDAL = "~0.10.12"
CairoMakie = "~0.15.14"
DataFrames = "~1.8.2"
Distances = "~0.10.12"
Distributions = "~0.25.131"
GeoDataFrames = "~0.4.4"
GeoInterface = "~1.6.2"
GeometryOps = "~0.1.46"
HiGHS = "~1.25.2"
JuMP = "~1.31.2"
LaTeXStrings = "~1.4.1"
PairPlots = "~3.0.8"
PlutoUI = "~0.7.83"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.13.0"
manifest_format = "2.1"
project_hash = "f725e9fbc4890bae42a880153a82cf529209fa1b"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
registries = "General"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
git-tree-sha1 = "e71ee7b4aa06b045259a7d6101e1cb45ad140bce"
registries = "General"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.4.1"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
registries = "General"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "7063ad1083578215c7c4bf410368150abe8d5524"
registries = "General"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.45"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "daa72978cd7a624246e894a4f4f067706d4e17e2"
registries = "General"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.7.0"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AdaptivePredicates]]
git-tree-sha1 = "7e651ea8d262d2d74ce75fdf47c4d63c07dba7a6"
registries = "General"
uuid = "35492f91-a3bd-45ad-95db-fcad7dcfedb7"
version = "1.2.0"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
registries = "General"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.Animations]]
deps = ["Colors"]
git-tree-sha1 = "e092fa223bf66a3c41f9c022bd074d916dc303e7"
registries = "General"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.2"

[[deps.ArchGDAL]]
deps = ["CEnum", "ColorTypes", "Dates", "DiskArrays", "Extents", "GDAL", "GeoFormatTypes", "GeoInterface", "ImageCore", "Tables"]
git-tree-sha1 = "d77f9c3d7bc0df0c7d197bc68a78a39664f1523d"
registries = "General"
uuid = "c9ce4bd3-c3d5-55b8-8973-c0e20141b8c3"
version = "0.10.12"

    [deps.ArchGDAL.extensions]
    ArchGDALJLD2Ext = "JLD2"
    ArchGDALMakieExt = "Makie"
    ArchGDALRecipesBaseExt = "RecipesBase"

    [deps.ArchGDAL.weakdeps]
    JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Arrow_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Lz4_jll", "Thrift_jll", "Zlib_jll", "Zstd_jll", "boost_jll", "brotli_jll", "snappy_jll"]
git-tree-sha1 = "55ecf3d16295c26e96d2f0b65386d1a8414e2283"
registries = "General"
uuid = "8ce61222-c28f-5041-a97a-c2198fb817bf"
version = "19.0.1+0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Automa]]
deps = ["PrecompileTools", "TranscodingStreams"]
git-tree-sha1 = "94eab0b3ccdcac361188cc661daf69d4433c1818"
registries = "General"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "1.2.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
registries = "General"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "4126b08903b777c88edf1754288144a0492c05ad"
registries = "General"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.8"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BaseDirs]]
git-tree-sha1 = "8c290a1b223deaeea9aea44b235d24546da8eb98"
registries = "General"
uuid = "18cc8868-cbac-4acf-b575-c8ff214dc66f"
version = "1.4.0"

[[deps.Blosc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Lz4_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "535c80f1c0847a4c967ea945fca21becc9de1522"
registries = "General"
uuid = "0b7ba130-8d10-5ba8-a3d6-c5182647fed9"
version = "1.21.7+0"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
registries = "General"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
registries = "General"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CRC32c]]
uuid = "8bf52ea8-c179-5cab-976a-9e18b702a9bc"
version = "1.11.0"

[[deps.CRlibm]]
deps = ["CRlibm_jll"]
git-tree-sha1 = "66188d9d103b92b6cd705214242e27f5737a1e5e"
registries = "General"
uuid = "96374032-68de-5a5b-8d9e-752f78720389"
version = "1.0.2"

[[deps.CRlibm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e329286945d0cfc04456972ea732551869af1cfc"
registries = "General"
uuid = "4e9b3aee-d8a1-5a3d-ad8b-7d824db253f0"
version = "1.0.1+0"

[[deps.Cairo]]
deps = ["Cairo_jll", "Colors", "Glib_jll", "Graphics", "Libdl", "Pango_jll"]
git-tree-sha1 = "71aa551c5c33f1a4415867fe06b7844faadb0ae9"
registries = "General"
uuid = "159f3aea-2a34-519c-b102-8c37f9878175"
version = "1.1.1"

[[deps.CairoMakie]]
deps = ["CRC32c", "Cairo", "Cairo_jll", "Colors", "FileIO", "FreeType", "GeometryBasics", "LinearAlgebra", "Makie", "PrecompileTools"]
git-tree-sha1 = "3495bfc164949714579501b825b8e5e2cce7c56f"
registries = "General"
uuid = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
version = "0.15.14"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1fa950ebc3e37eccd51c6a8fe1f92f7d86263522"
registries = "General"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.7+0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "12177ad6b3cad7fd50c8b3825ce24a99ad61c18f"
registries = "General"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CodecBzip2]]
deps = ["Bzip2_jll", "TranscodingStreams"]
git-tree-sha1 = "84990fa864b7f2b4901901ca12736e45ee79068c"
registries = "General"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.8.5"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "970758a3d591a2a5c2a907c53f2e2f8c1b1d3537"
registries = "General"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.9"

[[deps.CodecZstd]]
deps = ["TranscodingStreams", "Zstd_jll"]
git-tree-sha1 = "da54a6cd93c54950c15adf1d336cfd7d71f51a56"
registries = "General"
uuid = "6b39b394-51ab-5f42-8807-6242bab2b4c2"
version = "0.8.7"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON"]
git-tree-sha1 = "07da79661b919001e6863b81fc572497daa58349"
registries = "General"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
registries = "General"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
registries = "General"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
registries = "General"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
registries = "General"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.CommonSolve]]
deps = ["PrecompileTools"]
git-tree-sha1 = "6c389fa857f6ca5a95474b52a52023fd77f24cb7"
registries = "General"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.14"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
registries = "General"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
registries = "General"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.5.5+2"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
registries = "General"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ComputePipeline]]
deps = ["Observables", "Preferences"]
git-tree-sha1 = "7bc84b769c1d384315e7b5c4ac03a6c303e6cf35"
registries = "General"
uuid = "95dc2771-c249-4cd0-9c9f-1f3b4330693c"
version = "0.1.8"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
registries = "General"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"
weakdeps = ["IntervalSets", "LinearAlgebra", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
registries = "General"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.CoordinateTransformations]]
deps = ["LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "a692f5e257d332de1e554e4566a4e5a8a72de2b2"
registries = "General"
uuid = "150eb455-5306-5404-9cee-2592286d6298"
version = "0.6.4"

[[deps.CoreMath]]
deps = ["CoreMath_jll"]
git-tree-sha1 = "8c0480f92b1b1796239156a1b9b1bfb1b39499b4"
registries = "General"
uuid = "b7a15901-be09-4a0e-87d2-2e66b0e09b5a"
version = "0.1.0"

[[deps.CoreMath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a692a4c1dc59a4b8bc0b6403876eb3250fde2bc3"
registries = "General"
uuid = "a38c48d9-6df1-5ac9-9223-b6ada3b5572b"
version = "0.1.0+0"

[[deps.Crayons]]
git-tree-sha1 = "54b76cbb40d9a0f5368c880725b2f141da77c94f"
registries = "General"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.2.0"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
registries = "General"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "5fab31e2e01e70ad66e3e24c968c264d1cf166d6"
registries = "General"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.8.2"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "b0bc6d2cad1fed8b7fd59a1551a991cb3d2809e6"
registries = "General"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.6"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
registries = "General"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DelaunayTriangulation]]
deps = ["AdaptivePredicates", "EnumX", "ExactPredicates", "Random"]
git-tree-sha1 = "4ac548adcad90c1d5d677af13568a748af4c952b"
registries = "General"
uuid = "927a84f5-c5f4-47a5-9785-b46e178433df"
version = "1.6.7"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
registries = "General"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "79a2aca180a85c690c58a020d47b426954b590f8"
registries = "General"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.16.0"

[[deps.DiskArrays]]
deps = ["ConstructionBase", "LRUCache", "Mmap", "OffsetArrays"]
git-tree-sha1 = "9903195c34a488c5265d9a105f39718b7a2b3568"
registries = "General"
uuid = "3c3547ce-8d99-4f5e-a174-61eb10b00ae3"
version = "0.4.22"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "c7e3a542b999843086e2f29dac96a618c105be1d"
registries = "General"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.12"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "Roots", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "a958ab3a40c755563f5e1405c0846cb0446bf19d"
registries = "General"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.131"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsSparseConnectivityTracerExt = "SparseConnectivityTracer"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    SparseConnectivityTracer = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
registries = "General"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
registries = "General"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.EnumX]]
git-tree-sha1 = "c49898e8438c828577f04b92fc9368c388ac783c"
registries = "General"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.7"

[[deps.ExactPredicates]]
deps = ["IntervalArithmetic", "Random", "StaticArrays"]
git-tree-sha1 = "83231673ea4d3d6008ac74dc5079e77ab2209d8f"
registries = "General"
uuid = "429591f6-91af-11e9-00e2-59fbe8cec110"
version = "2.2.9"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2bfb1e047e2ad0a5ca94365340bde8005d637568"
registries = "General"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.8.4+0"

[[deps.Extents]]
git-tree-sha1 = "b309b36a9e02fe7be71270dd8c0fd873625332b4"
registries = "General"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.6"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libva_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "7a58e45171b63ed4782f2d36fdee8713a469e6e0"
registries = "General"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "8.1.2+0"

[[deps.FFTA]]
deps = ["AbstractFFTs", "DocStringExtensions", "LinearAlgebra", "MuladdMacro", "Primes", "Random", "Reexport"]
git-tree-sha1 = "65e55303b72f4a567a51b174dd2c47496efeb95a"
registries = "General"
uuid = "b86e33f2-c0db-4aa1-a6e0-ab43e668529e"
version = "0.3.1"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "6621fef488e496356c9c9625d0562c12a6070819"
registries = "General"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.20.0"

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

    [deps.FileIO.weakdeps]
    HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"

[[deps.FilePaths]]
deps = ["FilePathsBase", "MacroTools", "Reexport"]
git-tree-sha1 = "a1b2fbfe98503f15b665ed45b3d149e5d8895e4c"
registries = "General"
uuid = "8fc22ac5-c921-52a6-82fd-178b2807b824"
version = "0.9.0"

    [deps.FilePaths.extensions]
    FilePathsGlobExt = "Glob"
    FilePathsURIParserExt = "URIParser"
    FilePathsURIsExt = "URIs"

    [deps.FilePaths.weakdeps]
    Glob = "c27321d9-0574-5035-807b-f59d2c89b15c"
    URIParser = "30578b45-9adc-5946-b283-645ec420af67"
    URIs = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "3bab2c5aa25e7840a4b065805c0cdfc01f3068d2"
registries = "General"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.24"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "5bad39456d9f0166184fce2248783dd9862645c1"
registries = "General"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.17.0"
weakdeps = ["PDMats", "SparseArrays", "StaticArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStaticArraysExt = "StaticArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Random", "Statistics"]
git-tree-sha1 = "59af96b98217c6ef4ae0dfe065ac7c20831d1a84"
registries = "General"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.6"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
registries = "General"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
registries = "General"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "3b0f72e2ffef1a139ac450725c9ecd01c0a3c050"
registries = "General"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "1.4.6"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "907369da0f8e80728ab49c1c7e09327bf0d6d999"
registries = "General"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.1.1"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "70329abc09b886fd2c5d94ad2d9527639c421e3e"
registries = "General"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.14.3+1"

[[deps.FreeTypeAbstraction]]
deps = ["BaseDirs", "ColorVectorSpace", "Colors", "FreeType", "GeometryBasics", "Mmap"]
git-tree-sha1 = "4ebb930ef4a43817991ba35db6317a05e59abd11"
registries = "General"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.10.8"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
registries = "General"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GDAL]]
deps = ["CEnum", "GDAL_jll", "NetworkOptions", "PROJ_jll"]
git-tree-sha1 = "3eb19662b42c9d1c707b5ebe636595511c8b453d"
registries = "General"
uuid = "add2ef01-049f-52c4-9ee2-e494f65e021a"
version = "1.12.0"

[[deps.GDAL_jll]]
deps = ["Arrow_jll", "Artifacts", "Blosc_jll", "Expat_jll", "GEOS_jll", "HDF4_jll", "HDF5_jll", "JLLWrappers", "LERC_jll", "LibCURL_jll", "LibPQ_jll", "Libdl", "Libtiff_jll", "Lz4_jll", "MPIABI_jll", "MPICH_jll", "MPIPreferences", "MPItrampoline_jll", "MicrosoftMPI_jll", "NetCDF_jll", "OpenJpeg_jll", "OpenMPI_jll", "PCRE2_jll", "PROJ_jll", "Qhull_jll", "SQLite_jll", "TOML", "XML2_jll", "XZ_jll", "Zlib_jll", "Zstd_jll", "libgeotiff_jll", "libpng_jll", "libwebp_jll", "muparser_jll"]
git-tree-sha1 = "de59f28fe5c9133193e8ddeb2f4f52b42ccf0177"
registries = "General"
uuid = "a7073274-a066-55f0-b90d-d619367d196c"
version = "304.1200.400+0"

[[deps.GEOS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fdaf62d2354bb398652ee612d487eb19d74468a6"
registries = "General"
uuid = "d604d12d-fa86-5845-992e-78dc15976526"
version = "3.14.1+0"

[[deps.Gamma]]
deps = ["LogExpFunctions"]
git-tree-sha1 = "becc397f7cfb06e343496ae6ffb04818a851da51"
registries = "General"
uuid = "a0844989-3bd2-4988-8bea-c9407ab0941b"
version = "1.2.0"

[[deps.GeoDataFrames]]
deps = ["ArchGDAL", "DataAPI", "DataFrames", "Extents", "GeoFormatTypes", "GeoInterface", "GeometryOps", "Proj", "Reexport", "Tables", "WellKnownGeometry"]
git-tree-sha1 = "8eaad9327a15d8dec420e30abf784370fb3e2bd1"
registries = "General"
uuid = "62cb38b5-d8d2-4862-a48e-6a340996859f"
version = "0.4.4"

    [deps.GeoDataFrames.extensions]
    GeoDataFramesCSVExt = "CSV"
    GeoDataFramesDimensionalDataExt = "DimensionalData"
    GeoDataFramesFlatGeobufExt = "FlatGeobuf"
    GeoDataFramesGeoArrowExt = "GeoArrow"
    GeoDataFramesGeoJSONExt = "GeoJSON"
    GeoDataFramesGeoParquetExt = "GeoParquet"
    GeoDataFramesShapefileExt = "Shapefile"

    [deps.GeoDataFrames.weakdeps]
    CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
    DimensionalData = "0703355e-b756-11e9-17c0-8b28908087d0"
    FlatGeobuf = "d985ece1-97de-4d33-914c-38fb84042e15"
    GeoArrow = "5bc3a8d9-1bfb-4624-ba94-a391279174d6"
    GeoJSON = "61d90e0f-e114-555e-ac52-39dfb47a3ef9"
    GeoParquet = "e99870d8-ce00-4fdd-aeee-e09192881159"
    Shapefile = "8e980c4a-a4fe-5da2-b3a7-4b4b0353a2f4"

[[deps.GeoFormatTypes]]
git-tree-sha1 = "7528a7956248c723d01a0a9b0447bf254bf4da52"
registries = "General"
uuid = "68eda718-8dee-11e9-39e7-89f7f65f511f"
version = "0.4.5"

[[deps.GeoInterface]]
deps = ["DataAPI", "Extents", "GeoFormatTypes"]
git-tree-sha1 = "20f6b13b28c6304104968374f38c24ef71a8ea16"
registries = "General"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.6.2"

    [deps.GeoInterface.extensions]
    GeoInterfaceMakieExt = ["Makie", "GeometryBasics"]
    GeoInterfaceRecipesBaseExt = "RecipesBase"

    [deps.GeoInterface.weakdeps]
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "LinearAlgebra", "PrecompileTools", "Random", "StaticArrays"]
git-tree-sha1 = "592cfb5ed8b02804f6a9c04091571c393081f73a"
registries = "General"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.5.12"
weakdeps = ["Extents", "GeoInterface", "IntervalSets"]

    [deps.GeometryBasics.extensions]
    ExtentsExt = "Extents"
    GeometryBasicsGeoInterfaceExt = "GeoInterface"
    IntervalSetsExt = "IntervalSets"

[[deps.GeometryOps]]
deps = ["AbstractTrees", "AdaptivePredicates", "CoordinateTransformations", "DataAPI", "DelaunayTriangulation", "ExactPredicates", "Extents", "GeoFormatTypes", "GeoInterface", "GeometryOpsCore", "LinearAlgebra", "PrecompileTools", "Random", "SortTileRecursiveTree", "StaticArrays", "Statistics", "Tables"]
git-tree-sha1 = "f751768687839353286527489ab0279b09455e9d"
registries = "General"
uuid = "3251bfac-6a57-4b6d-aa61-ac1fef2975ab"
version = "0.1.46"

    [deps.GeometryOps.extensions]
    GeometryOpsDataFramesExt = "DataFrames"
    GeometryOpsDimensionalDataExt = "DimensionalData"
    GeometryOpsFlexiJoinsExt = "FlexiJoins"
    GeometryOpsLibGEOSExt = "LibGEOS"
    GeometryOpsMakieExt = "Makie"
    GeometryOpsProjExt = "Proj"
    GeometryOpsTGGeometryExt = "TGGeometry"

    [deps.GeometryOps.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    DimensionalData = "0703355e-b756-11e9-17c0-8b28908087d0"
    FlexiJoins = "e37f2e79-19fa-4eb7-8510-b63b51fe0a37"
    LibGEOS = "a90b1aa1-3769-5649-ba7e-abc5a9d163eb"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    Proj = "c94c279d-25a6-4763-9509-64d165bea63e"
    TGGeometry = "d7e755d2-3c95-4bcf-9b3c-79ab1a78647b"

[[deps.GeometryOpsCore]]
deps = ["DataAPI", "GeoInterface", "StableTasks", "Tables"]
git-tree-sha1 = "8f32a94cf3d80a716cae6025af6cbc4e68421da5"
registries = "General"
uuid = "05efe853-fabf-41c8-927e-7063c8b9f013"
version = "0.1.12"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
registries = "General"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Giflib_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6570366d757b50fabae9f4315ad74d2e40c0560a"
registries = "General"
uuid = "59f7168a-df46-5410-90c8-f2779963d0ec"
version = "5.2.3+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "090526e65de8f69648ac156daae153de8b56df62"
registries = "General"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.88.3+0"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "a641238db938fff9b2f60d08ed9030387daf428c"
registries = "General"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.3"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "69ffb934a5c5b7e086a0b4fee3427db2556fba6e"
registries = "General"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.16+0"

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "ef70da5e123a06a29e2d6ddff0f09985bc226491"
registries = "General"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.11.3"

[[deps.HDF4_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll", "libaec_jll"]
git-tree-sha1 = "ea9eff9cfef5f45b771096e5c2de3de0eab937c3"
registries = "General"
uuid = "818ab7a1-5177-5f44-ba99-6e845030c6cb"
version = "4.3.2+0"

[[deps.HDF5_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LibCURL_jll", "Libdl", "MPIABI_jll", "MPICH_jll", "MPIPreferences", "MPItrampoline_jll", "MicrosoftMPI_jll", "OpenMPI_jll", "OpenSSL_jll", "TOML", "Zlib_jll", "aws_c_s3_jll", "dlfcn_win32_jll", "libaec_jll", "mpif_jll"]
git-tree-sha1 = "45337643a2d97262d5fe72ce1f13e8a662d13d62"
registries = "General"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "2.1.2+0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "9d9531a9cb63a9edc33836414e82a07e81710de2"
registries = "General"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "100.14004.0+0"

[[deps.HiGHS]]
deps = ["HiGHS_jll", "LinearAlgebra", "MathOptIIS", "MathOptInterface", "OpenBLAS32_jll", "PrecompileTools", "SparseArrays"]
git-tree-sha1 = "9efeab5bba4fa60bc22de16a116efd730d1c7a17"
registries = "General"
uuid = "87dc4568-4c63-4d18-b0c0-bb2238e4078b"
version = "1.25.2"

[[deps.HiGHS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Zlib_jll", "libblastrampoline_jll"]
git-tree-sha1 = "d4e63f395d10590fcece8d35005dd7b9a4862635"
registries = "General"
uuid = "8fd58aa0-07eb-5a78-9b36-339c94fd15ea"
version = "1.15.1+3"

[[deps.Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XML2_jll", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "c35847ca5b4997fc8418836354a56c459bcf48d8"
registries = "General"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.14.0+0"

[[deps.HypergeometricFunctions]]
deps = ["Gamma", "LinearAlgebra"]
git-tree-sha1 = "31bb6c92405c084617facc1d7ed9eb6c402d061e"
registries = "General"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.30"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
registries = "General"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "d1a86724f81bcd184a38fd284ce183ec067d71a0"
registries = "General"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "1.0.0"

[[deps.ICU_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b3d8be712fbf9237935bde0ce9b5a736ae38fc34"
registries = "General"
uuid = "a51ab1cf-af8e-5615-a023-bc2c838bba6b"
version = "76.2.0+0"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
registries = "General"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "e12629406c6c4442539436581041d372d69c55ba"
registries = "General"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.12"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
registries = "General"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "8c193230235bbcee22c8066b0374f63b5683c2d3"
registries = "General"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.5"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs", "WebP"]
git-tree-sha1 = "f0f005f997dfb8c5fe23920d99458a9619873893"
registries = "General"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.10"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "2a81c3897be6fbcde0802a0ebe6796d0562f63ec"
registries = "General"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.10"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "dcc8d0cd653e55213df9b75ebc6fe4a8d3254c65"
registries = "General"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.2.2+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
registries = "General"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
registries = "General"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.InlineStrings]]
git-tree-sha1 = "06b65886c7577a3784d616e29f1302c2e36e389d"
registries = "General"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.6"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "c72458f1962faeb003bf23cbdb75164fe6280906"
registries = "General"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.4"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "48922d06068130f87e43edef52382e6a94305ae6"
registries = "General"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.16.3"
weakdeps = ["ForwardDiff", "Unitful"]

    [deps.Interpolations.extensions]
    InterpolationsForwardDiffExt = "ForwardDiff"
    InterpolationsUnitfulExt = "Unitful"

[[deps.IntervalArithmetic]]
deps = ["CRlibm", "CoreMath", "MacroTools", "OpenBLASConsistentFPCSR_jll", "Printf", "Random", "RoundingEmulator"]
git-tree-sha1 = "1c531bf0f8a5c60a340926e058fd3f209b5eef5d"
registries = "General"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "1.0.12"

    [deps.IntervalArithmetic.extensions]
    IntervalArithmeticArblibExt = "Arblib"
    IntervalArithmeticDiffRulesExt = "DiffRules"
    IntervalArithmeticForwardDiffExt = "ForwardDiff"
    IntervalArithmeticIntervalSetsExt = "IntervalSets"
    IntervalArithmeticIrrationalConstantsExt = "IrrationalConstants"
    IntervalArithmeticLinearAlgebraExt = "LinearAlgebra"
    IntervalArithmeticMakieExt = "Makie"
    IntervalArithmeticRecipesBaseExt = "RecipesBase"
    IntervalArithmeticSparseArraysExt = "SparseArrays"

    [deps.IntervalArithmetic.weakdeps]
    Arblib = "fb37089c-8514-4489-9461-98f9c8763369"
    DiffRules = "b552c78f-8df3-52c6-915a-8e097449b14b"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    IrrationalConstants = "92d709cd-6900-40b7-9082-c6be49f344b6"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.IntervalSets]]
git-tree-sha1 = "79d6bd28c8d9bccc2229784f1bd637689b256377"
registries = "General"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.14"

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

    [deps.IntervalSets.weakdeps]
    Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
registries = "General"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
registries = "General"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
registries = "General"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
registries = "General"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
registries = "General"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
registries = "General"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7204148362dafe5fe6a273f855b8ccbe4df8173e"
registries = "General"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.8.0"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "88352712893ec50bee3680605891eaf0e9ed6368"
registries = "General"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.8.0"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "9496de8fb52c224a2e3f9ff403947674517317d9"
registries = "General"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.6"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "037babc10853eeb8e585418922246cb97b8e5b74"
registries = "General"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.2.0+1"

[[deps.JuMP]]
deps = ["LinearAlgebra", "MacroTools", "MathOptInterface", "MutableArithmetics", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays"]
git-tree-sha1 = "4f27b21df3b47e8c08a83ead049afb621b2f5b3c"
registries = "General"
uuid = "4076af6c-e467-56ae-b986-b466b2749572"
version = "1.31.2"

    [deps.JuMP.extensions]
    JuMPDimensionalDataExt = "DimensionalData"

    [deps.JuMP.weakdeps]
    DimensionalData = "0703355e-b756-11e9-17c0-8b28908087d0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.Kerberos_krb5_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0f2899fdadaab4b8f57db558ba21bdb4fb52f1f0"
registries = "General"
uuid = "b39eb1a6-c29a-53d7-8c32-632cd16f18da"
version = "1.21.3+0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTA", "Interpolations", "StatsBase"]
git-tree-sha1 = "9eda8292dd3268b3b7ec9df21bbfac24e177ec52"
registries = "General"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.12"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
registries = "General"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "39bca05343661c347aae0bca57a5994a0bf4f08d"
registries = "General"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.2.0+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e5b100780d4d30d63b4618d7930d48af409c1772"
registries = "General"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "23.1.1+0"

[[deps.LRUCache]]
git-tree-sha1 = "5519b95a490ff5fe629c4a7aa3b3dfc9160498b3"
registries = "General"
uuid = "8ac3fa9e-de4c-5943-b1dc-09c6b5f20637"
version = "1.6.2"
weakdeps = ["Serialization"]

    [deps.LRUCache.extensions]
    SerializationExt = ["Serialization"]

[[deps.LaTeXStrings]]
git-tree-sha1 = "f88f3ccef05a6a72a0cf0ed417c8fd68530f4ab2"
registries = "General"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.1"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
registries = "General"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "1.0.0"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "Zstd_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.18.0+1"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "PCRE2_jll", "Zlib_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.1+0"

[[deps.LibPQ_jll]]
deps = ["Artifacts", "ICU_jll", "JLLWrappers", "Kerberos_krb5_jll", "Libdl", "OpenSSL_jll", "Zstd_jll"]
git-tree-sha1 = "c692057e05ba6da348bc45d5dab8c7a2c88da518"
registries = "General"
uuid = "08be9ffa-1c94-5ee5-a977-46a84ec9b350"
version = "16.14.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl", "OpenSSL_jll", "Zlib_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.103+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
registries = "General"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
registries = "General"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
registries = "General"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cc3ad4faf30015a3e8094c9b5b7f19e85bdf2386"
registries = "General"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.42.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "aebd334d06cee9f24cea70bd19a39749daf73881"
registries = "General"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.3+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d620582b1f0cbe2c72dd1d5bd195a9ce73370ab1"
registries = "General"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.42.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.13.0"

[[deps.LittleCMS_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll"]
git-tree-sha1 = "38928f7999753af13d4e13966ae15958ff3a917a"
registries = "General"
uuid = "d3a379c0-f9a3-5b72-a4c0-6bf4d2e8af0f"
version = "2.19.1+0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "bba2d9aa057d8f126415de240573e86a8f39d2a1"
registries = "General"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "1.0.1"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.Lz4_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "191686b1ac1ea9c89fc52e996ad15d1d241d1e33"
registries = "General"
uuid = "5ced341a-0733-55b8-9ab6-a4889d929147"
version = "1.10.1+0"

[[deps.MCMCDiagnosticTools]]
deps = ["AbstractFFTs", "DataAPI", "DataStructures", "Distributions", "LinearAlgebra", "MLJModelInterface", "Random", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "345bedeaf7d650f673fefa042dd8384c63de6c68"
registries = "General"
uuid = "be115224-59cd-429b-ad48-344e309966f0"
version = "0.3.19"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
registries = "General"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MLJModelInterface]]
deps = ["InteractiveUtils", "REPL", "Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "c275fae2e693206b4527dd9d2382aa15359ef3ed"
registries = "General"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.12.1"

[[deps.MPIABI_jll]]
deps = ["Artifacts", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "9be143b6045719e8fb019d2b3bc2aebad1184fef"
registries = "General"
uuid = "b5ada748-db0f-5fc0-8972-9331c762740c"
version = "0.1.5+0"

[[deps.MPICH_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "07dbec8aab01696edc0151a401a6cdfe95b9b885"
registries = "General"
uuid = "7cb0a576-ebde-5e09-9194-50597f1243b4"
version = "5.0.1+0"

[[deps.MPIPreferences]]
deps = ["Libdl", "Preferences"]
git-tree-sha1 = "8e98d5d80b87403c311fd51e8455d4546ba7a5f8"
registries = "General"
uuid = "3da0fdf6-3ccc-4f1b-acd9-58baa6c99267"
version = "0.1.12"

[[deps.MPItrampoline_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "675df097f8eeb28998b2cfe3b25655af73d5f7df"
registries = "General"
uuid = "f1f71cc9-e9ae-5b93-9b94-4fe0e1ad3748"
version = "5.5.6+0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
registries = "General"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Makie]]
deps = ["Animations", "Base64", "CRC32c", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "ComputePipeline", "Contour", "Dates", "DelaunayTriangulation", "Distributions", "DocStringExtensions", "Downloads", "FFMPEG_jll", "FileIO", "FilePaths", "FixedPointNumbers", "Format", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageBase", "ImageIO", "InteractiveUtils", "Interpolations", "IntervalSets", "InverseFunctions", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MacroTools", "Markdown", "MathTeXEngine", "Observables", "OffsetArrays", "PNGFiles", "Packing", "Pkg", "PlotUtils", "PolygonOps", "PrecompileTools", "Printf", "REPL", "Random", "RelocatableFolders", "Scratch", "ShaderAbstractions", "SignedDistanceFields", "SparseArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "TriplotBase", "UnicodeFun", "Unitful"]
git-tree-sha1 = "37b10d17f74f54dc5fa7d3c6c20fd75613c71d80"
registries = "General"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.24.14"

    [deps.Makie.extensions]
    MakieDynamicQuantitiesExt = "DynamicQuantities"

    [deps.Makie.weakdeps]
    DynamicQuantities = "06fc5a27-2a28-4c7c-a15d-362465fb6821"

[[deps.MappedArrays]]
git-tree-sha1 = "0ee4497a4e80dbd29c058fcee6493f5219556f40"
registries = "General"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.3"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MathOptIIS]]
deps = ["MathOptInterface"]
git-tree-sha1 = "3b3d69130d8ab8c39d5fa4d30e20a8e6428c9d37"
registries = "General"
uuid = "8c4f8055-bd93-4160-a86b-a0c04941dbff"
version = "0.2.0"

[[deps.MathOptInterface]]
deps = ["CodecBzip2", "CodecZlib", "ForwardDiff", "JSON", "LinearAlgebra", "MutableArithmetics", "NaNMath", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays", "SpecialFunctions", "Test"]
git-tree-sha1 = "d10ba577e0b5a0481fab01dfd31fb20af3326954"
registries = "General"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "1.53.0"

    [deps.MathOptInterface.extensions]
    MathOptInterfaceBenchmarkToolsExt = "BenchmarkTools"
    MathOptInterfaceCliqueTreesExt = "CliqueTrees"

    [deps.MathOptInterface.weakdeps]
    BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
    CliqueTrees = "60701a23-6482-424a-84db-faee86b9b1f8"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "UnicodeFun"]
git-tree-sha1 = "aa1078778be5a8e5259ff04fbc3d258b3e78d464"
registries = "General"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.6.9"

[[deps.Measures]]
git-tree-sha1 = "b513cedd20d9c914783d8ad83d08120702bf2c77"
registries = "General"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.3"

[[deps.MicrosoftMPI_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bc95bf4149bf535c09602e3acdf950d9b4376227"
registries = "General"
uuid = "9237b28f-5490-5468-be7b-bb81f5f5e6cf"
version = "10.1.4+3"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
registries = "General"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
registries = "General"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2026.8.13"

[[deps.MuladdMacro]]
deps = ["PrecompileTools"]
git-tree-sha1 = "283bf85d4a767481dd924dff0eee1735e95f449e"
registries = "General"
uuid = "46d2c3a1-f734-5fdb-9937-b9b9aeba4221"
version = "0.2.7"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "dc5b2c4c111c46bc79ac4405eeb563523b39c004"
registries = "General"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.8.0"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "dbd2e8cd2c1c27f0b584f6661b4309609c5a685e"
registries = "General"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.4"

[[deps.NetCDF_jll]]
deps = ["Artifacts", "Blosc_jll", "Bzip2_jll", "HDF5_jll", "JLLWrappers", "LazyArtifacts", "LibCURL_jll", "Libdl", "MPIABI_jll", "MPICH_jll", "MPIPreferences", "MPItrampoline_jll", "MicrosoftMPI_jll", "OpenMPI_jll", "TOML", "XML2_jll", "Zlib_jll", "Zstd_jll", "libaec_jll", "libzip_jll"]
git-tree-sha1 = "b9584045c11c1c89d24083bb654c37b73a447877"
registries = "General"
uuid = "7243133f-43d8-5620-bbf4-c2c921802cf3"
version = "401.1000.100+0"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
registries = "General"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
registries = "General"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "117432e406b5c023f665fa73dc26e79ec3630151"
registries = "General"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.17.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
registries = "General"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OpenBLAS32_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "libblastrampoline_jll"]
git-tree-sha1 = "30870d0f2dc0b2dba76b10df1c58c7f018413e56"
registries = "General"
uuid = "656ef2d0-ae68-5445-9ca0-591084a874a2"
version = "0.3.34+0"

[[deps.OpenBLASConsistentFPCSR_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "38a93f17e431141c6470bb67a88952a7c4f0e928"
registries = "General"
uuid = "6cdc7f73-28fd-5e50-80fb-958a8875b1af"
version = "0.3.34+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.30+0"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "97db9e07fe2091882c765380ef58ec553074e9c7"
registries = "General"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.3"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "1bcebd887dd33f1108210b3954049b1bb8af0e7a"
registries = "General"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.4.15+0"

[[deps.OpenJpeg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libtiff_jll", "LittleCMS_jll", "libpng_jll"]
git-tree-sha1 = "215a6666fee6d6b3a6e75f2cc22cb767e2dd393a"
registries = "General"
uuid = "643b3616-a352-519d-856d-80112ee9badc"
version = "2.5.5+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenMPI_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML", "Zlib_jll"]
git-tree-sha1 = "6d6c0ca4824268c1a7dca1f4721c535ac63d9074"
registries = "General"
uuid = "fe0851c0-eecd-5654-98d4-656369965a5c"
version = "5.0.11+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.6+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
registries = "General"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e2bb57a313a74b8104064b7efd01406c0a50d2ff"
registries = "General"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.6.1+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05f45c2e0de6259db764adbfd2f1dc6d3f8de13c"
registries = "General"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "2.0.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.46.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "123266c25174ef6c8d4718920abc206452cf8de6"
registries = "General"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.41"
weakdeps = ["StatsBase"]

    [deps.PDMats.extensions]
    StatsBaseExt = "StatsBase"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "32b657a0d57c310a1a172bfc8c8cf68c5e674323"
registries = "General"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.5"

[[deps.PROJ_jll]]
deps = ["Artifacts", "JLLWrappers", "LibCURL_jll", "Libdl", "Libtiff_jll", "SQLite_jll"]
git-tree-sha1 = "fbfbc14815c5f5375abc971321aa5f468e715a38"
registries = "General"
uuid = "58948b4f-47e0-5654-a9ad-f609743f8632"
version = "902.800.100+0"

[[deps.Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "bc5bf2ea3d5351edf285a06b0016788a121ce92c"
registries = "General"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.5.1"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
registries = "General"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.PairPlots]]
deps = ["Contour", "Distributions", "KernelDensity", "LinearAlgebra", "MCMCDiagnosticTools", "Makie", "Measures", "Missings", "OrderedCollections", "PolygonOps", "PrecompileTools", "Printf", "Requires", "StaticArrays", "Statistics", "StatsBase", "TableOperations", "Tables"]
git-tree-sha1 = "e4bf0aceaf2a9443c7c72ff59b1c2b954f210a32"
registries = "General"
uuid = "43a3c2be-4208-490b-832a-a21dcd55d7da"
version = "3.0.8"

    [deps.PairPlots.extensions]
    MCMCChainsExt = "MCMCChains"
    PairPlotsDynamicQuantitiesExt = "DynamicQuantities"
    PairPlotsDynamicUnitfulExt = "Unitful"

    [deps.PairPlots.weakdeps]
    DynamicQuantities = "06fc5a27-2a28-4c7c-a15d-362465fb6821"
    MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1912a9f1b9ca55005b03ba075f8e19993583e237"
registries = "General"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.58.2+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools"]
git-tree-sha1 = "663e8b48b789916221e0765393b289ca6c88f24e"
registries = "General"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "3.0.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "e4a6721aa89e62e5d4217c0b21bd714263779dda"
registries = "General"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.46.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "Zstd_jll", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.13.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
registries = "General"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "26ca162858917496748aad52bb5d3be4d26a228a"
registries = "General"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.4"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e189d0623e7ce9c37389bac17e80aac3b0302e75"
registries = "General"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.83"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
registries = "General"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
registries = "General"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "edbeefc7a4889f528644251bdb5fc9ab5348bc2c"
registries = "General"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "8b770b60760d4451834fe79dd483e318eee709c4"
registries = "General"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.2"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "REPL", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "1b8aa19f229b1cea7fc93874a52e49db6a854450"
registries = "General"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "3.4.8"

    [deps.PrettyTables.extensions]
    PrettyTablesExcelExt = "XLSX"
    PrettyTablesTypstryExt = "Typstry"

    [deps.PrettyTables.weakdeps]
    Typstry = "f0ed7684-a786-439e-b1e3-3b82803b501e"
    XLSX = "fdbf4ff8-1666-58a4-91e7-1b58723a45e0"

[[deps.Primes]]
deps = ["IntegerMathUtils"]
git-tree-sha1 = "25cdd1d20cd005b52fc12cb6be3f75faaf59bb9b"
registries = "General"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.7"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "fbb92c6c56b34e1a2c4c36058f68f332bec840e7"
registries = "General"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.11.0"

[[deps.Proj]]
deps = ["CEnum", "CoordinateTransformations", "GeoFormatTypes", "GeoInterface", "NetworkOptions", "PROJ_jll"]
git-tree-sha1 = "61188669db4f5b400173e4ec60da8bcb72d6e749"
registries = "General"
uuid = "c94c279d-25a6-4763-9509-64d165bea63e"
version = "1.9.0"

[[deps.PtrArrays]]
git-tree-sha1 = "4fbbafbc6251b883f4d2705356f3641f3652a7fe"
registries = "General"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.4.0"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "472daaa816895cb7aee81658d4e7aec901fa1106"
registries = "General"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.2"

[[deps.Qhull_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c69da20496799bbdd56c15ecf5d80a5e6cbcc904"
registries = "General"
uuid = "784f63db-0788-585a-bace-daefebcd302b"
version = "10008.0.1004+0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "5e8e8b0ab68215d7a2b14b9921a946fee794749e"
registries = "General"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.3"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["Base64", "Dates", "FileWatching", "InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
registries = "General"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
registries = "General"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
registries = "General"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
registries = "General"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
registries = "General"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "5b3d50eb374cea306873b371d3f8d3915a018f0b"
registries = "General"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.9.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6d40b2fe70437b01397d2a4d5b020008da4e7019"
registries = "General"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.2+0"

[[deps.Roots]]
deps = ["Accessors", "CommonSolve", "Printf"]
git-tree-sha1 = "4db094d5e079abbda658acfe1c4d098430417717"
registries = "General"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "3.0.8"

    [deps.Roots.extensions]
    RootsChainRulesCoreExt = "ChainRulesCore"
    RootsForwardDiffExt = "ForwardDiff"
    RootsIntervalRootFindingExt = "IntervalRootFinding"
    RootsSymPyExt = "SymPy"
    RootsSymPyPythonCallExt = "SymPyPythonCall"
    RootsUnitfulExt = "Unitful"

    [deps.Roots.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalRootFinding = "d2bf35a9-74e0-55ec-b149-d360ff49b807"
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"
    SymPyPythonCall = "bc8888f7-b21e-4b7c-a06a-5d9c9496438c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.RoundingEmulator]]
git-tree-sha1 = "40b9edad2e5287e05bd413a38f61a8ff55b9557b"
registries = "General"
uuid = "5eaf0fd0-dfba-4ccb-bf02-d820a40db705"
version = "0.2.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "1.0.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "e24dc23107d426a096d3eae6c165b921e74c18e4"
registries = "General"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.7.2"

[[deps.SQLite_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll", "dlfcn_win32_jll"]
git-tree-sha1 = "324744e40b84e6dc2cfaa4122ce969a083c40a6f"
registries = "General"
uuid = "76ed43ae-9a5d-5a62-8c75-30186b810ce8"
version = "3.53.2+0"

[[deps.ScientificTypesBase]]
deps = ["InteractiveUtils"]
git-tree-sha1 = "e785eaa35a0f5518a388f9010e66fda64ea95ede"
registries = "General"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "3.1.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
registries = "General"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "084c47c7c5ce5cfecefa0a98dff69eb3646b5a80"
registries = "General"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.10"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.ShaderAbstractions]]
deps = ["ColorTypes", "FixedPointNumbers", "GeometryBasics", "LinearAlgebra", "Observables", "StaticArrays"]
git-tree-sha1 = "818554664a2e01fc3784becb2eb3a82326a604b6"
registries = "General"
uuid = "65257c39-d410-5151-9873-9b3e5be5013e"
version = "0.5.0"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.SignedDistanceFields]]
deps = ["Statistics"]
git-tree-sha1 = "3949ad92e1c9d2ff0cd4a1317d5ecbba682f4b92"
registries = "General"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.1"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "7ddb0b49c109481b046972c0e4ab02b2127d6a75"
registries = "General"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.6"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "0494aed9501e7fb65daba895fb7fd57cc38bc743"
registries = "General"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.5"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortTileRecursiveTree]]
deps = ["AbstractTrees", "Extents", "GeoInterface"]
git-tree-sha1 = "f9aa6616a9b3bd01f93f27c010f1d25fc5a094a9"
registries = "General"
uuid = "746ee33f-1797-42c2-866d-db2fce69d14d"
version = "0.1.4"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "13cd91cc9be159e3f4d95b857fa2aa383b53772a"
registries = "General"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.3"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.13.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "429071b23f4c9a13fb6582f807cc2ef454082408"
registries = "General"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.9.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "4f96c596b8c8258cc7d3b19797854d368f243ddc"
registries = "General"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.4"

[[deps.StableTasks]]
git-tree-sha1 = "c4f6610f85cb965bee5bfafa64cbeeda55a4e0b2"
registries = "General"
uuid = "91464d47-22a1-43fe-8b7f-2d57ee82463f"
version = "0.1.7"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "be1cf4eb0ac528d96f5115b4ed80c26a8d8ae621"
registries = "General"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.2"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "e206cf4850fd7ac4255ffd2b98922f563e18ac53"
registries = "General"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.20"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
registries = "General"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "89f86d9376acd18a1a4fbef66a56335a3a7633b8"
registries = "General"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "3.5.0"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "e2b53ce13a53367e96601081e33d34746b571bad"
registries = "General"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.5"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
registries = "General"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "adb9da019510162e67a4493fc235c23203d8b09e"
registries = "General"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.13"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "91a5737baed20ee31f3faea0e51f57461f6a689e"
registries = "General"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "2.2.1"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "773065c6e0e903924a9d838259be74338422aef2"
registries = "General"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.5.0"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "ad8002667372439f2e3611cfd14097e03fa4bccd"
registries = "General"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.7.3"

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = ["GPUArraysCore", "KernelAbstractions"]
    StructArraysLinearAlgebraExt = "LinearAlgebra"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

    [deps.StructArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "2d0fc55c61321ba245c47be599570d11bac50303"
registries = "General"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.8.5"

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsStaticArraysCoreExt = ["StaticArraysCore"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.10.1+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
registries = "General"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
registries = "General"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "a94d9bdda1b7bed0046cea645639ab3f62196fac"
registries = "General"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.14.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
registries = "General"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.Thrift_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "boost_jll"]
git-tree-sha1 = "4d16a4b4eab80099c19342b10d0bdb252c39bea6"
registries = "General"
uuid = "e0b8ae26-5307-5830-91fd-398402328850"
version = "0.21.1+0"

[[deps.TiffImages]]
deps = ["CodecZstd", "ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "PrecompileTools", "ProgressMeter", "SIMD", "UUIDs"]
git-tree-sha1 = "9ca5f1f2d42f80df4b8c9f6ab5a64f438bbd9976"
registries = "General"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.11.9"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
registries = "General"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
registries = "General"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.TriplotBase]]
git-tree-sha1 = "4d4ed7f294cda19382ff7de4c137d24d16adc89b"
registries = "General"
uuid = "981d1d27-644d-49a2-9326-4793e63143c3"
version = "0.1.0"

[[deps.URIs]]
git-tree-sha1 = "908fec9df6c5de98548ead82a468c95ccf6cd263"
registries = "General"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.7.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
registries = "General"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "1f0f9f401753701a7e4113b5056ca38d33875b55"
registries = "General"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.29.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    ForwardDiffExt = "ForwardDiff"
    InverseFunctionsUnitfulExt = "InverseFunctions"
    LatexifyExt = ["Latexify", "LaTeXStrings"]
    NaNMathExt = "NaNMath"
    PrintfExt = "Printf"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"
    LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
    Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
    NaNMath = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
    Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.WebP]]
deps = ["CEnum", "ColorTypes", "FileIO", "FixedPointNumbers", "ImageCore", "libwebp_jll"]
git-tree-sha1 = "aa1ca3c47f119fbdae8770c29820e5e6119b83f2"
registries = "General"
uuid = "e3aaa7dc-3e4b-44e0-be63-ffb868ccd7c1"
version = "0.1.3"

[[deps.WellKnownGeometry]]
deps = ["GeoFormatTypes", "GeoInterface"]
git-tree-sha1 = "7954cccbdf5af33a677ad7e8a4436f073703fe61"
registries = "General"
uuid = "0f680547-7be7-4555-8820-bb198eeb646b"
version = "0.2.7"

    [deps.WellKnownGeometry.extensions]
    WellKnownGeometryMakieExt = "Makie"
    WellKnownGeometryRecipesBaseExt = "RecipesBase"

    [deps.WellKnownGeometry.weakdeps]
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "248a7031b3da79a127f14e5dc5f417e26f9f6db7"
registries = "General"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.1.0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "80d3930c6347cfce7ccf96bd3bafdf079d9c0390"
registries = "General"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.9+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e52eca002a11c30a858185efdfb15311e1c7a6bf"
registries = "General"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.4+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "808090ede1d41644447dd5cbafced4731c56bd2f"
registries = "General"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.13+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
registries = "General"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
registries = "General"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "1a4a26870bf1e5d26cd585e38038d399d7e65706"
registries = "General"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.8+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "75e00946e43621e09d431d9b95818ee751e6b2ef"
registries = "General"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.2+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
registries = "General"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libpciaccess_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "58972370b81423fc546c56a60ed1a009450177c3"
registries = "General"
uuid = "a65dc6b1-eb27-53a1-bb3e-dea574b5389e"
version = "0.19.0+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
registries = "General"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
registries = "General"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["CompilerSupportLibraries_jll", "Libdl"]
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.aws_c_auth_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_cal_jll", "aws_c_http_jll", "aws_c_sdkutils_jll"]
git-tree-sha1 = "8cab83c96af80a1be968251ce1a0548a7545484d"
registries = "General"
uuid = "2b3700d1-4306-52e2-a478-c162f0c514be"
version = "0.9.6+0"

[[deps.aws_c_cal_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_common_jll"]
git-tree-sha1 = "22c0f42f4a1f0dc5dcfa8fd267c4ac407c455e7a"
registries = "General"
uuid = "70f11efc-bab2-57f1-b0f3-22aad4e67c4b"
version = "0.9.13+0"

[[deps.aws_c_common_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a759cb9bf456ad792cc7898a81ae333cce9ef02a"
registries = "General"
uuid = "73048d1d-b8c4-5092-a58d-866c5e8d1e50"
version = "0.12.6+0"

[[deps.aws_c_compression_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_common_jll"]
git-tree-sha1 = "7910c72f45f44afd297c39fe43b99c56d5ed22ec"
registries = "General"
uuid = "73a04cd5-f3d7-5bac-9290-e8adb709f224"
version = "0.3.2+0"

[[deps.aws_c_http_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_compression_jll", "aws_c_io_jll"]
git-tree-sha1 = "3fb8685778068de502c72fec5dd8075e037cee15"
registries = "General"
uuid = "3254fc65-9028-534d-aa9d-d76d128babc6"
version = "0.10.15+0"

[[deps.aws_c_io_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_cal_jll", "aws_c_common_jll", "s2n_tls_jll"]
git-tree-sha1 = "7e481d474b2087ee8bbf55b81bf9119f21e396d9"
registries = "General"
uuid = "13c41daa-f319-5298-b5eb-5754e0170d52"
version = "0.26.3+0"

[[deps.aws_c_s3_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_auth_jll", "aws_c_common_jll", "aws_c_http_jll", "aws_checksums_jll", "s2n_tls_jll"]
git-tree-sha1 = "3e9917ab25114feba657e71be41cad068b9f6595"
registries = "General"
uuid = "bd1f34fb-993f-5903-a121-aaf302eed6d4"
version = "0.11.5+0"

[[deps.aws_c_sdkutils_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_common_jll"]
git-tree-sha1 = "c43dfba2c1ab9ea9f02f2c80e86fa16f6460244e"
registries = "General"
uuid = "1282aa60-004d-510b-9f52-12498d409daa"
version = "0.2.4+1"

[[deps.aws_checksums_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "aws_c_common_jll"]
git-tree-sha1 = "2570c8e23f4771a087b12a47edcaaa670ac05a01"
registries = "General"
uuid = "b2a88e68-78e7-5e94-8c20-c02986ec140e"
version = "0.2.10+0"

[[deps.boost_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "25fb6ecbb784a45f8ea74584fa631a9e85393dd0"
registries = "General"
uuid = "28df3c45-c428-5900-9ff8-a3135698ca75"
version = "1.87.0+0"

[[deps.brotli_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "46fda47f4215c957bc92fd5fbb5ad04fee1e3743"
registries = "General"
uuid = "4611771a-a7d2-5e23-8d00-b1becdba1aae"
version = "1.2.0+0"

[[deps.dlfcn_win32_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e141d67ffe550eadfb5af1bdbdaf138031e4805f"
registries = "General"
uuid = "c4b69c83-5512-53e3-94e6-de98773c479f"
version = "1.4.2+0"

[[deps.isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
registries = "General"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

[[deps.libaec_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "60f4792734488db6f42e2c7699f1d4594780bd03"
registries = "General"
uuid = "477f73a3-ac25-53e9-8cc3-50b2fa2566f0"
version = "1.1.7+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ef17c47d22224aaecc76e597ab21a072e025cf7b"
registries = "General"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.14.1+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "cb007192783c56d8249db4cf0e3495001edfe414"
registries = "General"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.5+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libdrm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "28e57478e8a160d346a19c28b3fffb9273bcc9c2"
registries = "General"
uuid = "8e53e030-5e6c-5a89-a30b-be5b7263a166"
version = "2.4.134+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
registries = "General"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libgeotiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LibCURL_jll", "Libdl", "Libtiff_jll", "PROJ_jll", "Zlib_jll"]
git-tree-sha1 = "cbdbc9ae1127f81cb653a4f7545d89f8db2a17a7"
registries = "General"
uuid = "06c338fa-64ff-565b-ac2f-249532af990e"
version = "100.702.400+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e51150d5ab85cee6fc36726850f0e627ad2e4aba"
registries = "General"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.58+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "libpng_jll"]
git-tree-sha1 = "c1733e347283df07689d71d61e14be986e49e47a"
registries = "General"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.5+0"

[[deps.libva_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "libdrm_jll"]
git-tree-sha1 = "7dbf96baae3310fe2fa0df0ccbb3c6288d5816c9"
registries = "General"
uuid = "9a156e7d-b971-5f62-b2c9-67348b8fb97c"
version = "2.23.0+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
registries = "General"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.libwebp_jll]]
deps = ["Artifacts", "Giflib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libglvnd_jll", "Libtiff_jll", "libpng_jll"]
git-tree-sha1 = "4e4282c4d846e11dce56d74fa8040130b7a95cb3"
registries = "General"
uuid = "c5f90fcd-3b7e-5836-afba-fc50a0988cb2"
version = "1.6.0+0"

[[deps.libzip_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "OpenSSL_jll", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "363a1c43b8cb92e7b3ff7f504ef9a0a83c548e97"
registries = "General"
uuid = "337d8026-41b4-5cde-a456-74a10e5b31d1"
version = "1.11.4+0"

[[deps.mpif_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIABI_jll", "MPICH_jll", "MPIPreferences", "MPItrampoline_jll", "MicrosoftMPI_jll", "OpenMPI_jll", "TOML"]
git-tree-sha1 = "a8083ee0737c243c8f40a4ba86a0956997facb73"
registries = "General"
uuid = "9aeb927a-4695-514f-a259-621a69f20ec0"
version = "0.1.7+0"

[[deps.muparser_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "70ee0f42a44ef6e16298e5bfc8b6e311d08e49bb"
registries = "General"
uuid = "888e69b1-873b-5047-a2fc-24c07cbe9dc8"
version = "2.3.5+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.67.1+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.8.2+0"

[[deps.s2n_tls_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a968513a5d3b3f3c6e9cbc73edb0d9c05b9fe77e"
registries = "General"
uuid = "cddc5d3d-934d-5d3a-9747-62fc12ea3f48"
version = "1.7.9+0"

[[deps.snappy_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ca88363dd41d2547f52118287dd34dbbc14f3eb7"
registries = "General"
uuid = "fe1e1685-f7be-5f59-ac9f-4ca204017dfd"
version = "1.2.3+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
registries = "General"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
registries = "General"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"

[registries.General]
url = "https://github.com/JuliaRegistries/General.git"
uuid = "23338594-aafe-5451-b93e-139f81909106"
"""

# ╔═╡ Cell order:
# ╠═bf822e2a-f5a1-49f6-9a3d-2a326e85facd
# ╟─3a979d09-3d6d-4398-85a2-c380f70a170d
# ╟─300bf605-8503-4f0e-86c0-9616c7c863fe
# ╟─b2a3713d-e71f-405a-b84d-83b66a703ffc
# ╟─cfc8c30d-7dd2-4899-9fe7-678e6476d1c0
# ╟─ac4d4716-94f6-410d-a136-f8ffabbb63f0
# ╟─a1c85d82-1ac6-4981-97d9-c3afab3ee599
# ╟─c1a566f0-9650-40aa-818e-707a141e77b6
# ╟─3aae6e9b-54c3-40ce-a411-65a60879c713
# ╟─e664545c-abbe-4c7a-b2b2-c14614ecaa12
# ╟─c73bbb70-56fc-4f2a-9a0f-38666bcd1faa
# ╟─d95deda5-a9d8-45fc-8ec7-4c8fdffb8638
# ╟─59e6d996-3a62-4b02-8774-be424307de15
# ╟─8f5dc258-762b-4bfa-a673-9478016c5e4a
# ╟─b6a06a94-b6fe-45d3-9d1d-3a7fbc8a90bc
# ╟─58f9c253-314a-4154-8e24-7cc7637809da
# ╟─48edb189-23ad-411f-be62-05d5c669fd9f
# ╟─c007daea-26d6-4f23-8aeb-1a0db6af1091
# ╠═ec2dd1c0-6613-4cbd-97fb-5c89fc95186d
# ╟─bd6b1a6d-c7bf-4c67-b962-aff24fe66eaf
# ╟─e8df9819-1a23-488d-b41b-3ffc8ca537d6
# ╠═432f5b73-f506-4ac8-af8e-0a30de194c94
# ╠═61a1ab2f-5ad8-412e-a6c6-496b9f37f284
# ╟─6fe45813-cfd7-4a7a-b052-6f98980d2298
# ╟─b44bd516-7e49-4e7b-a6e8-b91166799158
# ╠═ab206d1d-6702-4f2a-b11a-f17066bc6cb1
# ╠═559b59e7-b6d8-4ae3-8957-cc98c55dc372
# ╟─5a9b18f7-12ee-49b6-88e8-8fa1e93bef8f
# ╟─ef41ea3f-c771-49b0-aabf-99dc00196302
# ╟─30f18612-dcd2-455e-9df0-e5aaa95d4852
# ╠═829f8e9e-aa43-48cb-bb15-9150e0a4ff69
# ╟─926b411d-40bd-424f-8074-40bf4fbd0985
# ╠═2cf97702-5204-4605-9da5-3e130be4dfff
# ╠═4ea83ba8-85d9-4962-b20a-176593649e0d
# ╟─3b318522-990a-44d2-979b-29457e28df0b
# ╠═b073829d-3b5f-49db-9ede-f3c4cf7c9b86
# ╟─6af65f24-fdff-47cf-9d25-8736c4e43a9d
# ╟─1f7cb1bb-a5e4-486e-9e9f-ad7d4757e34e
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
