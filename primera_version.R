
import::from(magrittr, "%$%", ex = extract, .into = "operadores")
import::from(zeallot, `%<-%`)
pacman::p_load(DBI, dbplyr, conectigo, janitor, tidyverse)

limpiar <- function(x) {

	str_sub(x, 1, 24) |>
		str_trim(side = "both") |>
		str_remove_all(pattern = "[[:punct:]]+") |>
		str_squish()

}


con <- conectar_msql()


ttks_00 <- tbl(con, in_schema("tkd", "w_ttks")) |>
	collect()

afectacion_00 <- tbl(con, in_schema("tkd", "w_afectacion")) |>
	collect()

descargas_00 <- tbl(con, in_schema("md", "descarga_vista")) |>
	filter(estado_descarga_item %in% c("aplicado", "pendiente")) |>
	select(fecha = fecha_uso,
								cantidad,
								contratista,
								ticketid = ticket,
								precio = usd,
								item = nombre_informal) |>
	collect()

materiales_01 <- descargas_00 |>
	filter(contratista != "Masscardy") |>
	select(ticketid,
								item,
								precio,
								cantidad) |>
	mutate(across(precio, na_if, "NULL"),
								across(precio, parse_number),
								total = precio * cantidad) |>
	relocate(precio, cantidad) |>
	group_by(ticketid) |>
	summarise(materiales = sum(cantidad),
											monto = sum(total),
											.groups = "drop") |>
	arrange(desc(monto)) |>
	drop_na()


ttks_01 <- ttks_00 |>
	filter(
		estado        == "CERRADO",
		aplica        == "SI",
		documentacion == "SI",
		tipo_mtto     == "CORRECTIVO"
		) |>
	mutate(
		lugar = case_when(
			topografia == "EN_CLIENTE"  ~ "INTERIOR",
			TRUE ~ "EXTERIOR"),
		lugar = fct_reorder(lugar, ttr, .desc = F)) |>
	select(ticketid, ttr, zona, lugar, nivel1, nivel2, nivel3, bu) |>
	drop_na()


ext <- ttks_01 |>
	filter(
		# nivel1 == "FIBRA_OPTICA",
		lugar == "EXTERIOR")
		ttr > quantile(ttr, 0.15)


ttks_01 |>
	filter(
		lugar == "INTERIOR") |> count(nivel3) |> print(n = Inf)

int <- ttks_01 |>
	filter(
		lugar == "INTERIOR",
		# ttr < quantile(ttr, 0.065),
		# nivel3 %in% c("CORTE_PATCHCORD", "EQUIPO_APAGADO_CLIENTE")
)

place_00 <- bind_rows(ext, int)


afectacion_01 <- afectacion_00 |>
	group_by(ticketid) |>
	summarise(servicios = n()) |>
	arrange(desc(servicios))



place_01 <- place_00 |>
	inner_join(afectacion_01, by = "ticketid") |>
	inner_join(materiales_01, by = "ticketid") |>
	# mutate(across(detalle, limpiar)) |>
	select(lugar, ttr, servicios, materiales, monto, zona, bu)


place_01 |> tabyl(lugar)

int_s <- place_01 |>
	filter(lugar == "INTERIOR") |>
	slice_sample(n = 40)

ext_s <- place_01 |>
	filter(lugar == "EXTERIOR")

place_02 <- bind_rows(int_s, ext_s)


place_02 |> tabyl(lugar)

# agregar un poco de ruido
place_01$lugar[sample(nrow(place_01), 40)] <- "INTERIOR"


place_01 |> tabyl(lugar)


write_csv(x = place_02, file = "place.csv")



