archivo_sobrevivio = "3\\gender_submission.csv"
archivo_test = "3\\test.csv"
archivo_entrenamiento = "3\\train.csv"


def generar_csv_completo(archivo_gender, archivo_test, archivo_train):

    archivo_gender = archivo_gender
    archivo_test = archivo_test
    archivo_train = archivo_train

    def unirSurv_test(archivo_gender, archivo_test):
        # Datos sobrevivio e id:
        sobrevivio = {}
        with open(archivo_gender, mode="r", encoding="utf-8") as f:
            datos = f.readlines()
            for i in datos:
                valores = i.replace("\n", "").split(",")
                sobrevivio[valores[0]] = valores[1]
        f.close()

        with open(archivo_test, mode="r", encoding="utf-8-sig") as file:
            resultado = []
            datos_2 = file.readlines()

            for i, items in enumerate(datos_2):
                valores_2 = items.strip().split(",")

                if i == 0:
                    valores_2.append("Survived")
                else:
                    valores_2.append(sobrevivio.get(valores_2[0]))
                resultado.append(valores_2)
            file.close()

        with open("data_con_sobrevivientes.csv", mode="w", encoding="utf-8") as fi:
            for fila in resultado:
                fi.write(",".join(fila) + "\n")
        return True

    def unificar_csvs(archivo_train):
        archivo_train = archivo_train

        def leer_csv(archivo):
            with open(archivo, mode="r", encoding="utf-8-sig") as file:
                lineas = file.readlines()

            # Separar encabezado y datos
            encabezados = lineas[0].strip().split(",")
            datos = [line.strip().split(",") for line in lineas[1:]]

            return encabezados, datos

        encabezados_train, datos_train = leer_csv(archivo_train)
        encabezados_data_con, datos_data_con = leer_csv(
            "data_con_sobrevivientes.csv")

        # survived al final
        idx_survived = encabezados_train.index("Survived")
        encabezados_train.pop(idx_survived)
        encabezados_train.append("Survived")

        # Reordenar "Survived" en train
        datos_train_reordenado = []
        for fila in datos_train:
            survived = fila.pop(idx_survived)
            fila.append(survived)
            datos_train_reordenado.append(fila)

        if "Survived" not in encabezados_data_con:
            # Agregar "Survived" al final si no está
            encabezados_data_con.append("Survived")

        # Insertar valores de survived
        datos_data_con_modificado = []
        idx_survived_data = encabezados_data_con.index("Survived")

        for fila in datos_data_con:
            if len(fila) <= idx_survived_data or fila[idx_survived_data] == "":
                fila.append("2")  # Si falta el valor, agregar "2"
            datos_data_con_modificado.append(fila)

        # unir datos
        encabezados_final = encabezados_train
        datos_finales = datos_train_reordenado + datos_data_con_modificado

        archivo_final = "archivo_unificado.csv"
        # guardar archivo
        with open(archivo_final, mode="w", encoding="utf-8") as file:
            file.write(",".join(encabezados_final) + "\n")
            for fila in datos_finales:
                file.write(",".join(fila) + "\n")

        return archivo_final

    def insertar_valores(archivo_entrada):
        archivo_salida = "3\\data_completa.csv"

        with open(archivo_entrada, mode="r", encoding="utf-8-sig") as file:
            lineas = file.readlines()

        # Leer encabezados
        encabezados = lineas[0].strip().split(",")

        # Procesar datos y agregar a la columna vacia
        datos_filtrados = []

        # generar media para la edad y despues agregarle la media a los que estan vacios
        n = 0
        valor = 0

        for linea in lineas[1:]:
            valores = linea.strip().split(",")
            if valores[5] != "":
                if "." in valores[5]:
                    valor += int(float(valores[5]))
                else:
                    valor += int(valores[5])
                n += 1
                # print(valor, n)
        media_edad = round((valor/n))
        # print(media_edad)

        for linea in lineas[1:]:
            valores = linea.strip().split(",")

            if valores[5] == "":  # edad vacia
                valores.insert(5, str(float(media_edad)))
                valores.pop(6)
            else:
                valor_original_edad = valores[5]
                valores.insert(5, str(float(valor_original_edad)))
                valores.pop(6)

            if valores[10] == "":  # cabina desconocida
                valores.insert(10, valores[10].replace("", "DES"))
                valores.pop(11)
            else:
                valor_original_cabina = valores[10]
                valores.insert(10, str(valor_original_cabina))
                valores.pop(11)

            datos_filtrados.append(valores)

        # Guardar el archivo
        with open(archivo_salida, mode="w", encoding="utf-8") as file:
            file.write(",".join(encabezados) + "\n")
            for fila in datos_filtrados:
                file.write(",".join(fila) + "\n")

        return True

    unirSurv_test(archivo_gender, archivo_test)
    insertar_valores(unificar_csvs(archivo_train))
    return True


generar_csv_completo(archivo_sobrevivio, archivo_test, archivo_entrenamiento)
