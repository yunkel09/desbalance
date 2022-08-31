

#   ____________________________________________________________________________
#   proyecto: evaluar impacto del desbalance de clases                      ####


##  ............................................................................
##  paquetes                                                                ####

	import::from(magrittr, "%$%", ex = extract, .into = "operadores")
	import::from(zeallot, `%<-%`)
	pacman::p_load(DBI, dbplyr, conectigo, janitor, tidyverse)


##  ............................................................................
##  cargar                                                                  ####

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


##  ............................................................................
##  prep                                                                    ####

	ttks_01 <- ttks_00 |>
		filter(
			estado == "CERRADO",
			aplica == "SI",
			documentacion == "SI") |>
		mutate(
			interior = case_when(
				topografia == "EN_CLIENTE"  ~ 1L,
				TRUE ~ 0L)) |>
		select(ticketid, ttr, zona, nivel3, bu, interior) |>
		drop_na()

	exterior <- ttks_01 |>
		filter(
			interior == 0
			# ttr > quantile(ttr, 0.15)
		)

	interior <- ttks_01 |>
		filter(
			interior == 1,
			ttr < quantile(ttr, 0.065),
			nivel3 %in% c("CORTE_PATCHCORD", "EQUIPO_APAGADO_CLIENTE")
		)

	ttks_03 <- bind_rows(interior, exterior)

	ttks_04 <- ttks_03 |>
		inner_join(afectacion_01, by = "ticketid") |>
		inner_join(materiales_01, by = "ticketid") |>
		select(-ticketid) |>
		rename(categoria = nivel3) |>
		relocate(interior, ttr, servicios, materiales, monto, zona, bu, categoria)


##  ............................................................................
##  evaluar proporción                                                      ####

	ttks_04 |> tabyl(interior)


##  ............................................................................
##  guardar                                                                 ####

	write_csv(x = ttks_04, file = "interior.csv")

#   ____________________________________________________________________________
#   fin_script                                                              ####


