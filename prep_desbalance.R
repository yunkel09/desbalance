
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


afectacion_00 <- tbl(con, in_schema("tkd", "w_afectacion")) |>
	collect()

afectacion_01 <- afectacion_00 |>
	group_by(ticketid) |>
	summarise(servicios = n()) |>
	arrange(desc(servicios))


ttks_00 <- tbl(con, in_schema("tkd", "w_ttks")) |>
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
		estado == "CERRADO",
		aplica == "SI") |>
	select(ticketid, topografia, ttr, zona, nivel2) |>
	drop_na()


ttks_02 <- ttks_01 |>
	mutate(
		interior = case_when(
			topografia == "EN_CLIENTE"  ~ 1L,
			TRUE ~ 0L)) |>
	select(-topografia)


ttks_03 <- ttks_02 |>
	inner_join(afectacion_01, by = "ticketid") |>
	select(interior, ttr, zona, servicios, categoria = nivel2)

int_s <- ttks_03 |>
	filter(interior == 1) |>
	slice_sample(n = 88)

ext_s <- ttks_03 |>
	filter(interior == 0)

ttks_04 <- bind_rows(int_s, ext_s)

ttks_04 |> tabyl(interior)

ttks_05 <- ttks_04 |>
	select(interior, ttr)

write_csv(x = ttks_04, file = "interior.csv")



