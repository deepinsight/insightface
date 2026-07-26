# Guia de utilização da API REST do InsightFace Server

**Idiomas:** [English](api.md) · [中文](api.zh-CN.md) · [日本語](api.ja.md) · [Deutsch](api.de.md) · [Español](api.es.md) · [Français](api.fr.md) · [Русский](api.ru.md) · Português · [한국어](api.ko.md)

Este guia descreve o objetivo, entrada, trabalho do servidor, resultado e erros de cada API pública. Para instalação e primeira pesquisa consulte o [guia do utilizador](user-guide.pt.md); o esquema exato da instância está em `/docs` e `/openapi.json`.

## Regras comuns

- Base `/v1`, JSON em `snake_case`, imagens JPEG/PNG/WebP em multipart.
- O Compose fornecido desativa autenticação para avaliação isolada. Quando ativa, tudo exceto health requer `Authorization: Bearer <api_key>`; quando inativa omita completamente o cabeçalho.
- Cada resposta tem `x-request-id`; JSON repete-o como `request_id`.
- confidence/quality/threshold usam `0..1`. Similarity não é probabilidade: é cosine bruto `[-1,1]`; predefinição `0.4`, match quando `similarity >= threshold`.
- Cursor é opaco e só deve ser devolvido sem alterações à mesma rota, Collection, Person e filtro.
- Estados comuns: 400 entrada, 401 auth, 404 ausente, 409 conflito, 413 tamanho, 422 imagem/rosto, 429 limite, 503 timeout/modelo/índice.

```bash
BASE_URL=http://127.0.0.1:18097
AUTH_HEADER="Authorization: Bearer ${INSIGHTFACE_API_KEY}"
curl -fsS "${BASE_URL}/v1/health"
```

## Sistema

### `GET /v1/health`

**Uso/entrada:** readiness pública, sem parâmetros nem auth. **Resultado:** verifica arranque e SQLite quick_check; 200 com `status`, `auth_enabled`, `request_id`. **Erro:** `503 not_ready`.

### `GET /v1/system`

**Uso/entrada:** diagnóstico seguro, sem parâmetros. **Resultado:** 200 com OS/CPU/GPU, Driver, CUDA/cuDNN/ORT, Provider, modelo, DB, mounts, contagens, pesquisa, configuração segura e concorrência; nunca segredos, imagens ou embeddings. **Erros:** 401, 503.

### `GET /v1/models`

**Uso/entrada:** detector/recognizer verificados, Provider e licença; sem parâmetros. **Resultado:** 200 `models`, `execution_provider`, `license`. **Erro:** 401.

## Operações faciais sem estado

### `POST /v1/detect`

**Entrada:** multipart `image` obrigatório, `max_faces` 1–100, `collection_id` opcional. **Processo/resultado:** combina resoluções, NMS global, ordena por área; 200 `faces` com caixas/5 pontos/score/qualidade e `processing_ms`. Sem rosto é lista vazia válida. **Erros:** 400 min_score antigo, 404 Collection, 413, 422 invalid_image, 503.

```bash
curl -sS "${BASE_URL}/v1/detect" -H "${AUTH_HEADER}" -F 'image=@group.jpg' -F 'max_faces=10'
```

### `POST /v1/compare`

**Entrada:** multipart `source`, `target`, `threshold` opcional 0–1 e `collection_id`. **Resultado:** escolhe um rosto por imagem; 200 `matched`, cosine `similarity`, threshold efetivo, ambos os rostos e tempo. **Erros:** 404, 413, 422 invalid_image/face_not_found, 503.

### `POST /v1/embeddings`

**Entrada:** multipart `image`, `collection_id` opcional. **Resultado:** 200 com rosto escolhido, embedding L2, modelo e tempo. Desnecessário para registo normal; vetor não é registado em logs. **Erros:** 400 face_selection antigo, 404, 413, 422, 503.

## Collections

### `POST /v1/collections`

**Entrada:** JSON `id`, `name`; opcionais description, threshold (0.4), metadata, save_face_crops, `detection`, `search` com profile/capacity/max_faces_per_person/load_policy. **Processo/resultado:** fixa modelo, pré-processamento e contrato de pesquisa; 201 com `collection` resolvida. **Erros:** 400 configuração, 409 exists, 503 índice.

```bash
curl -sS "${BASE_URL}/v1/collections" -H "${AUTH_HEADER}" -H 'Content-Type: application/json' -d '{"id":"employees","name":"Employees","threshold":0.4}'
```

### `GET /v1/collections`

**Entrada:** query `limit` 1–100 (50), cursor opcional. **Resultado:** 200 `collections`, `next_cursor` nullable. **Erros:** 400 invalid_cursor, 401.

### `GET /v1/collections/{collection_id}`

**Entrada:** ID da Collection no path. **Resultado:** 200 `collection`, contagens Person/Face e `embedding_contract_id`. **Erro:** 404.

### `PATCH /v1/collections/{collection_id}`

**Entrada:** ID; JSON name/description/threshold/metadata/save_face_crops, capacidade/max/load de pesquisa e detection. Null, campos desconhecidos, modelo e search profile não podem mudar. **Resultado:** 200 Collection completa; detection vale no próximo pedido. **Erros:** 400, 404, 409, 503.

### `DELETE /v1/collections/{collection_id}`

**Entrada:** ID; query `force=false`, true se não vazia. **Resultado:** 204 sem corpo. **Erros:** 404, 409 collection_not_empty, 503.

## Pessoas e FaceSamples

### `POST /v1/collections/{collection_id}/persons`

**Entrada:** Collection; multipart `images` repetível, id/name/external_id opcionais, metadata como JSON texto, `review_mode=off|standard|strict`, `embedding_mode=server|external_trusted`; modo externo acrescenta vetores e contract ID. **Processo/resultado:** revê cada imagem; 201 `person`, `faces` aceites e `rejected_images`, sucesso parcial; tudo rejeitado dá 422 sem Person. **Erros:** 400, 404, 409 ID/contrato/capacidade, 413, 422, 503.

```bash
curl -sS "${BASE_URL}/v1/collections/employees/persons" -H "${AUTH_HEADER}" -F 'id=alice' -F 'review_mode=off' -F 'images=@alice.jpg'
```

### `GET /v1/collections/{collection_id}/persons`

**Entrada:** Collection; query limit/cursor/`search` por ID, nome ou external ID. **Resultado:** 200 `persons`, `next_cursor`. **Erros:** 400 cursor, 404.

### `GET /v1/collections/{collection_id}/persons/{person_id}`

**Entrada:** IDs Collection e Person. **Resultado:** 200 `person` com face_count. **Erro:** 404.

### `PATCH /v1/collections/{collection_id}/persons/{person_id}`

**Entrada:** IDs; JSON name/external_id/metadata objeto. **Resultado:** 200 Person atualizada. **Erros:** 400, 404, 409 external_id_exists.

### `DELETE /v1/collections/{collection_id}/persons/{person_id}`

**Entrada:** IDs. **Resultado:** elimina Person, FaceSamples, embeddings e crops, sincroniza índice, 204. **Erros:** 404, 503.

### `POST /v1/collections/{collection_id}/persons/{person_id}/faces`

**Entrada:** IDs; images repetíveis e os mesmos campos review/embedding da criação. **Resultado:** 201 `faces`, `rejected_images`, sucesso parcial. **Erros:** registo mais 404 Person.

### `GET /v1/collections/{collection_id}/persons/{person_id}/faces`

**Entrada:** IDs; query limit 1–100 e cursor. **Resultado:** 200 metadata `faces`, `has_crop`, `next_cursor`, sem embedding nem bytes. **Erros:** 400 cursor, 404.

### `GET /v1/collections/{collection_id}/persons/{person_id}/faces/{face_id}/image`

**Entrada:** três IDs. **Resultado:** se guardado, 200 `image/jpeg`, crop 112×112, `Cache-Control:no-store`; request ID apenas no header. **Erros:** 401, 404 face/face_image_not_found.

### `DELETE /v1/collections/{collection_id}/persons/{person_id}/faces/{face_id}`

**Entrada:** três IDs. **Resultado:** elimina embedding/crop/linha do índice, 204. **Erros:** 404, 503.

## Pesquisa

### `POST /v1/collections/{collection_id}/search`

**Entrada:** Collection; multipart `image`, `limit` 1–100 (5), threshold opcional ou valor da Collection. **Processo/resultado:** compara o rosto escolhido com todas as samples e usa o máximo por Person; 200 `searched_face`, `matches` ordenados, threshold e tempo. Sem match é lista vazia. **Erros:** 404, 409 modelo, 413, 422 imagem/rosto, 503 índice/timeout.

```bash
curl -sS "${BASE_URL}/v1/collections/employees/search" -H "${AUTH_HEADER}" -F 'image=@query.jpg' -F 'limit=5'
```

## Monitores RTSP

A configuração do Monitor persiste no SQLite e uma tarefa ativada é restaurada após reiniciar o servidor. Os fotogramas não são guardados; os eventos vivem apenas num buffer circular limitado em memória.

### `POST /v1/monitors`

**Uso:** Criar um Monitor RTSP persistente. **Entrada:** JSON com ID, nome, `source`, Collection, `inference_fps` (2), limiar opcional, buffer/política de eventos e `preview_enabled` (false). **Resultado:** 201 com `monitor` ocultado; credenciais são encriptadas em repouso. **Erros:** 400, 404, 409, 429.

### `GET /v1/monitors`

**Uso:** Listar Monitores com paginação. **Entrada:** `limit` 1–100 (50) e o `cursor` opaco da resposta anterior, sem alterações. **Resultado:** 200 com `monitors` e `next_cursor`, nunca as credenciais. **Erros:** 400 `invalid_cursor`, 401.

### `GET /v1/monitors/{monitor_id}`

**Uso:** Ler configuração e resumo de execução de um Monitor. **Entrada:** `monitor_id` no caminho. **Resultado:** 200 com política de eventos, origem ocultada, preview e estado. **Erros:** 401, 404 `monitor_not_found`.

### `PATCH /v1/monitors/{monitor_id}`

**Uso:** Atualizar parcialmente exceto o ID e iniciar/parar com `enabled`. **Entrada:** JSON parcial; `event_policy` também é parcial e limiar null herda a Collection. **Resultado:** 200 com Monitor completo; origem/Collection/frequência/política reiniciam a tarefa. **Erros:** 400, 404, 429.

### `DELETE /v1/monitors/{monitor_id}`

**Uso:** Remover permanentemente um Monitor. **Entrada:** `monitor_id` no caminho. **Resultado:** para descodificação, inferência e RTSP, descarta eventos de memória e devolve 204; não remove a Collection. **Erros:** 401, 404.

### `GET /v1/monitors/{monitor_id}/state`

**Uso:** Consultar o estado atual em clientes sem interface. **Entrada:** ID do Monitor. **Resultado:** 200 com ligação, FPS efetivo, latência, saltos, rostos reconhecidos/desconhecidos, preview, reconexões e erro seguro, sem embeddings. **Erros:** 401, 404.

### `GET /v1/monitors/{monitor_id}/events`

**Uso:** Obter eventos voláteis de entrada/saída/erro/recuperação. **Entrada:** `limit` 1–1000 e último `cursor` opaco. **Resultado:** 200 com `events`, `next_cursor`, `truncated` e `stream_reset`; reiniciar perde eventos. **Erros:** 400 `invalid_cursor`, 401, 404.

### `GET /v1/monitors/{monitor_id}/preview.mjpeg`

**Uso:** Abrir o preview MJPEG bruto, desativado por predefinição. **Entrada:** ID e cabeçalho Bearer normal; nunca a chave no URL. **Resultado:** `multipart/x-mixed-replace` longo, codificado só com observadores; o cliente desenha caixas via `/state`. **Erros:** 401, 404, 409 `preview_disabled`, 503.

## Repetição de pedidos

GET pode ser repetido. Verifique o estado antes de repetir DELETE. Se o resultado de criar Person/Face for incerto pela rede, consulte o ID antes de novo POST. Repita apenas 429 e 503 transitórios com backoff exponencial limitado e jitter; corrija os 4xx.
