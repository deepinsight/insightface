# Guia de utilização da API REST do InsightFace Server

**Idiomas:** [English](api.md) · [中文](api.zh-CN.md) · [日本語](api.ja.md) · [Deutsch](api.de.md) · [Español](api.es.md) · [Français](api.fr.md) · [Русский](api.ru.md) · Português · [한국어](api.ko.md)

Este guia descreve o objetivo, entrada, trabalho do servidor, resultado e erros de cada API pública. Para instalação e primeira pesquisa consulte o [guia do utilizador](user-guide.pt.md); o esquema exato da instância está em `/docs` e `/openapi.json`.


Os modelos são identificados por `model_id`; as respostas não incluem um `model_version` separado. As Collections existentes mantêm o seu `embedding_contract_id`.

Para usar a prova de vida, consulte [configuração, instalação e resultados](#addon-opcional-de-prova-de-vida). Cada operação explica também os seus efeitos.

## Regras comuns

- Base `/v1`, JSON em `snake_case`, imagens JPEG/PNG/WebP/BMP em multipart.
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

## Addon opcional de prova de vida

A prova de vida está desativada por predefinição em `server/config/server.toml`: `inference.addons` e `addons.auto_download` são `[]`. Configurações antigas sem estas chaves continuam desativadas.

**Ativar pela linha de comandos, incluindo antes do primeiro arranque de Server:**

```bash
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models install buffalo_l --accept-license --enable-liveness
```

`--enable-liveness` verifica primeiro se a configuração existente pode ser atualizada. Instala e verifica o pacote base, os addons configurados para instalação e a prova de vida, e depois acrescenta `liveness` a `inference.addons` e `addons.auto_download`, preservando outras entradas, comentários e definições. Reutiliza caches verificados, mas guarda a ativação mesmo quando os modelos já estão em cache. Uma descarga falhada não altera a configuração; uma falha ao guardar produz um erro explícito e um código de saída diferente de zero. Os ficheiros válidos em cache podem ser reutilizados numa nova tentativa.

Os dois serviços Compose montam todo o diretório existente `server/config` com escrita em `/etc/insightface`, com `create_host_path: false`. Assim o instalador atualiza atomicamente a configuração do host sem Server em execução. O diretório e `server.toml` têm de existir.

Server não precisa de estar em execução. Numa instalação nova, o próximo `up -d` ativa a prova de vida; se Server já estiver em execução, use `docker compose -f server/deploy/compose.cpu.yml restart server`. `up -d` sozinho não recarrega as definições guardadas. Para CUDA use `compose.cuda12.yml`.

Sem `--enable-liveness`, `models install` mantém o comportamento e não escreve configuração; a prova de vida continua desativada por predefinição. `models addons install liveness` apenas descarrega e verifica o addon, sem o ativar. Também pode ativar em **Sistema → Deteção de vida**, como descrito abaixo.

Em **Sistema → Detecção de vivacidade**, descarregue o modelo e ative-o para o próximo arranque. Após verificar SHA-256, acrescenta-se `liveness` às duas listas, preservando as outras entradas, comentários e opções. Um ficheiro verificado é reutilizado. **Reinicie manualmente o Server** para aplicar. Os erros permitem repetir; uma descarga falhada não ativa a prova de vida.

[Montagens e permissões para descargas Web](user-guide.pt.md#montagens-e-permissões-para-descargas-web).

**Avançado: configuração manual.** Estas definições são uma alternativa ao parâmetro de ativação ou à ação Web; instale o modelo antes de reiniciar.

```toml
[inference]
addons = ["liveness"]
liveness_mode = "normal"
liveness_threshold = 0.8
liveness_compare_scope = "both"
liveness_on_registration = false

[addons]
auto_download = ["liveness"]
```

### Instalação do modelo e arranque

`inference.addons` controla a execução e `addons.auto_download` a descarga adicional na instalação do pacote base. Com `["liveness"]`, o addon é instalado mesmo com o pacote base em cache. Não há descarga no arranque. Instalador e Server leem o mesmo ficheiro.

Execute os comandos na raiz do repositório com `server/config/server.toml` presente. O Compose fornecido executa o instalador como root, cria o diretório de modelos quando necessário e monta `/models` com escrita; a descarga do addon cria `addons`. Não é necessário preparar UID/GID ou permissões manualmente. Consulte a [configuração inicial do guia do utilizador](user-guide.pt.md).

```bash
docker compose -f server/deploy/compose.cpu.yml run --rm models addons install liveness
docker compose -f server/deploy/compose.cpu.yml run --rm models addons verify liveness
docker compose -f server/deploy/compose.cpu.yml up -d --force-recreate
```

Um modelo ativado ausente impede o arranque com `addon_model_missing`; um inválido produz `addon_model_invalid`. O addon não é desativado silenciosamente.

### Resultados da prova de vida

| Resultado | `status` | `is_live` | `live_score` |
| --- | --- | --- | --- |
| Rosto vivo | `ok` | `true` | `[0, 1]` |
| Falsificação | `ok` | `false` | `[0, 1]` |
| Entrada rejeitada | `input_rejected` | `null` | `null` |

Apenas uma área insuficiente da imagem original em redor do rosto alinhado produz `input_rejected`. Este resultado acrescenta `liveness.reason`, uma explicação para o utilizador; os resultados de rosto vivo ou falsificação omitem `reason`. FaceAnalysis e a API devolvem sempre este texto em inglês; só a interface Web traduz a apresentação. Use `status` e `is_live` na lógica do programa, sem interpretar o texto de `reason`. Resultados antigos guardados podem não incluir `reason`; o cliente pode então apresentar uma mensagem genérica de entrada rejeitada.

```json
{
  "status": "input_rejected",
  "is_live": null,
  "live_score": null,
  "reason": "Insufficient image area around the face for liveness detection. Move the face toward the center, step back from the camera, or use a less tightly cropped image."
}
```

`normal` reconhece apenas rostos aprovados; `observe` devolve o resultado e continua o reconhecimento. Sem avaliação, `liveness` é omitido. Os três campos principais são `status`, `is_live` e `live_score`: os resultados de rosto vivo ou falsificação usam `status: ok`, booleano e pontuação; entrada rejeitada usa `status: input_rejected` e dois valores `null`.

`/v1/detect` retorna HTTP 200 mesmo para resultados negativos. Em `normal`, embeddings, comparação e busca retornam HTTP 422 `liveness_fake` ou `liveness_input_rejected` com `error.details.liveness`; comparação acrescenta `details.side`. Falhas retornam HTTP 503 `liveness_unavailable`. As falhas de execução interrompem a operação tanto em `normal` como em `observe`; não são convertidas em `input_rejected`.

O cadastro de pessoas e a adição de FaceSamples ignoram a prova de vida por padrão: `[inference].liveness_on_registration=false` não executa o modelo e omite `liveness` nas novas amostras. Com `true` e o addon habilitado, aplica-se `normal`/`observe`; rejeições incluem `reason` e `liveness`. A revisão de qualidade por `review_mode` e a validação dos embeddings externos continuam ativas. `review_mode=off` e `external_trusted` não ignoram uma prova de cadastro habilitada. As requisições não podem alterar esta configuração de inicialização. Os resultados já salvos continuam disponíveis.

RTSP distingue `liveness_blocked` de `unknown` e conta `liveness_blocked_faces`. Rostos bloqueados não geram eventos de entrada de pessoas/desconhecidos e reiniciam a confirmação. Falhas de inferência apagam identidades exibidas anteriormente.

`liveness_compare_scope` escolhe os lados de `/v1/compare`: `both` (predefinido) avalia ambos, `source` a imagem de origem e `target` a imagem de destino. O rosto é considerado vivo se `live_score >= liveness_threshold`.

`models addons install liveness` guarda o modelo publicado em `/models/addons/liveness.onnx`; no anfitrião Compose, em `server/.models/addons/liveness.onnx`. Os erros de arranque são `addon_model_missing` e `addon_model_invalid`. `/v1/models` e `/v1/system` apresentam os complementos ativos em `addons`.

[Configuração e operações](user-guide.pt.md#addon-opcional-de-prova-de-vida).

## Sistema

### `GET /v1/health`

**Uso/entrada:** readiness pública, sem parâmetros nem auth. **Resultado:** verifica arranque e SQLite quick_check; 200 com `status`, `auth_enabled`, `request_id`. **Erro:** `503 not_ready`.

### `GET /v1/system`

**Uso/entrada:** diagnóstico seguro, sem parâmetros. **Resultado:** 200 com OS/CPU/GPU, Driver, CUDA/cuDNN/ORT, Provider, modelo, DB, mounts, contagens, pesquisa, configuração segura e concorrência; nunca segredos, imagens ou embeddings. **Erros:** 401, 503.

### `GET /v1/models`

`addons` apresenta os addons ativos separados do modelo base. Verifique `liveness` e as definições efetivas em `safe_config` da resposta de sistema. Estes endpoints são só de leitura e não instalam modelos.

**Uso/entrada:** detector/recognizer verificados, Provider e licença; sem parâmetros. **Resultado:** 200 `models`, `execution_provider`, `license`. **Erro:** 401.

Os pacotes base `raccoon_s` e `raccoon_l` suportam CPU e CUDA e são instalados com a ferramenta de modelos antes do arranque. Esta API lista os componentes em execução, não um catálogo de descargas. A ação Web abaixo gere apenas a prova de vida. As Collections estão vinculadas ao modelo de reconhecimento e ao pré-processamento: mudar o pacote não converte os vetores existentes e pode gerar `409 collection_model_mismatch`. Ativar apenas a prova de vida não altera esse contrato.

### `GET /v1/addons/liveness`

**Utilização:** Consultar a instalação e as definições do próximo arranque sem descarregar nem alterar nada. É uma API de gestão, não uma API independente de inferência de prova de vida.

**Resultado:** HTTP 200. `enabled` indica o estado do processo atual. `installed` significa que o ficheiro passou a verificação SHA-256 publicada, não que a prova está ativa. `configured_enabled` lê a seleção do próximo arranque no ficheiro atual; `restart_required` indica que difere de `enabled`. Até reiniciar, `safe_config` de `/v1/system` continua a descrever o processo atual.

`state` é `idle` (sem modelo verificado), `downloading` (preparação em curso), `ready` (modelo verificado disponível) ou `error` (erro de preparação, ficheiro ou configuração). `ready` por si só não confirma a gravação da ativação nem a conclusão do reinício.

`can_enable` indica se a preparação Web está disponível. Caso contrário, `unavailable_code` fornece um código estável do motivo e `unavailable_reason` uma explicação; nos restantes casos ambos são `null`. `error` é `null` ou um objeto com `code` e `message`. `model_path` é o caminho local do modelo; `config_file`, o caminho TOML selecionado ou `null`. A resposta também contém `request_id`.

Os valores de `unavailable_code` são `config_file_missing` (sem ficheiro de configuração selecionado), `config_file_not_regular` (não é um ficheiro normal), `config_file_mount` (ficheiro montado individualmente), `config_not_writable` (configuração sem permissão de escrita), `addon_directory_not_writable` (diretório do complemento sem escrita), `addon_config_invalid` (configuração inválida), `addon_model_invalid` (modelo inválido) e `server_stopping` (servidor a encerrar).

```json
{
  "enabled": false,
  "installed": true,
  "configured_enabled": true,
  "restart_required": true,
  "can_enable": true,
  "unavailable_code": null,
  "unavailable_reason": null,
  "state": "ready",
  "error": null,
  "model_path": "/models/addons/liveness.onnx",
  "config_file": "/etc/insightface/server.toml",
  "request_id": "3ed21e89-4595-4eed-a699-1df42ca62032"
}
```

### `POST /v1/addons/liveness/enable`

**Utilização:** Descarregar a prova de vida e configurar a ativação no próximo arranque. Enviar um objeto JSON vazio `{}` com `Content-Type: application/json`. Não são aceites URL de modelos nem outros parâmetros.

```bash
curl -sS "${BASE_URL}/v1/addons/liveness" -H "${AUTH_HEADER}"
curl -sS "${BASE_URL}/v1/addons/liveness/enable" -H "${AUTH_HEADER}" \
  -H 'Content-Type: application/json' -d '{}'
```

**Resultado:** HTTP 202 devolve os mesmos campos que GET e confirma a aceitação do trabalho, não a ativação. Consultar `GET /v1/addons/liveness` até terminar. Pedidos repetidos partilham o trabalho ativo; fechar o navegador não o cancela.

Só depois da descarga e verificação SHA-256 é acrescentado `liveness` a `[inference].addons` e `[addons].auto_download` em `config_file`, preservando outros valores e comentários. Um ficheiro verificado é reutilizado. Com `installed=true`, `configured_enabled=true` e `restart_required=true`, reiniciar manualmente o servidor. O novo processo apresenta `enabled=true` e `restart_required=false`. Não existe recarregamento em execução nem API para mudar o pacote base.

**Erros:** Os erros de pedido usam o formato habitual: `400 invalid_addon_request` se o corpo não for `{}`, `401 unauthorized` se a autenticação falhar, `403 origin_not_allowed` para uma origem de navegador não permitida, `409 addon_management_unavailable` para caminhos, permissões ou definições incompatíveis, e `415 json_required` se o conteúdo não for JSON. O navegador deve usar a mesma origem do servidor ou uma origem explicitamente permitida por CORS.

Um trabalho aceite pode falhar mais tarde: GET continua a devolver HTTP 200 com `state=error` e `error.code`. `addon_download_failed` não altera a configuração; verificar a rede ou o proxy do servidor. Para `addon_config_save_failed`, corrigir a configuração ou as permissões do diretório; o modelo verificado continua reutilizável. `addon_config_invalid` indica TOML inválido em disco. `addon_model_invalid` exige substituir ou remover o ficheiro inválido, que nunca é sobrescrito silenciosamente. `addon_job_in_progress` indica outro processo em preparação: aguardar e atualizar. Corrigir a causa antes de repetir POST.

## Operações faciais sem estado

### `POST /v1/detect`

Cada rosto avaliado inclui `liveness.status`, `liveness.is_live` e `liveness.live_score`. Os resultados de falsificação e `input_rejected` também devolvem HTTP 200, sem extrair características de reconhecimento. `input_rejected` indica uma área de imagem insuficiente em redor do rosto; `liveness.reason` explica como ajustar a imagem. Sem `liveness`, não houve avaliação.

**Entrada:** multipart `image` obrigatório, `max_faces` 1–100, `collection_id` opcional. **Processo/resultado:** combina resoluções, NMS global, ordena por área; 200 `faces` com caixas/5 pontos/score/qualidade e `processing_ms`. Sem rosto é lista vazia válida. **Erros:** 400 min_score antigo, 404 Collection, 413, 422 invalid_image, 503.

```bash
curl -sS "${BASE_URL}/v1/detect" -H "${AUTH_HEADER}" -F 'image=@group.jpg' -F 'max_faces=10'
```

### `POST /v1/compare`

`liveness_compare_scope` (`both`, `source`, `target`) escolhe os lados avaliados antes do reconhecimento. Em `normal`, a rejeição devolve HTTP 422 `liveness_fake` / `liveness_input_rejected`, `error.details.liveness` e `error.details.side`, sem similaridade. `observe` continua e inclui os resultados nos rostos avaliados.

**Entrada:** multipart `source`, `target`, `threshold` opcional 0–1 e `collection_id`. **Resultado:** escolhe um rosto por imagem; 200 `matched`, cosine `similarity`, threshold efetivo, ambos os rostos e tempo. **Erros:** 404, 413, 422 invalid_image/face_not_found, 503.

### `POST /v1/embeddings`

Com prova de vida em `normal`, uma falsificação ou uma entrada inadequada devolve HTTP 422 `liveness_fake` / `liveness_input_rejected` e `error.details.liveness`; não extrai o embedding. `observe` devolve o embedding e o resultado do rosto.

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

Criar Person e adicionar FaceSamples ignora a prova de vida por predefinição (`liveness_on_registration=false`). Se ativada, `normal` rejeita falsificações e entradas inadequadas; `observe` guarda o resultado e continua. A revisão de qualidade segue o `review_mode` selecionado. A lista de rejeições mostra o `reason` real e a prova de vida separadamente.

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

Criar Person e adicionar FaceSamples ignora a prova de vida por predefinição (`liveness_on_registration=false`). Se ativada, `normal` rejeita falsificações e entradas inadequadas; `observe` guarda o resultado e continua. A revisão de qualidade segue o `review_mode` selecionado. A lista de rejeições mostra o `reason` real e a prova de vida separadamente.

**Entrada:** IDs; images repetíveis e os mesmos campos review/embedding da criação. **Resultado:** 201 `faces`, `rejected_images`, sucesso parcial. **Erros:** registo mais 404 Person.

### `GET /v1/collections/{collection_id}/persons/{person_id}/faces`

**Entrada:** IDs; query limit 1–100 e cursor. **Resultado:** 200 metadata `faces`, `has_crop`, `next_cursor`, sem embedding nem bytes. **Erros:** 400 cursor, 404.

### `GET /v1/collections/{collection_id}/persons/{person_id}/faces/{face_id}/image`

**Entrada:** três IDs. **Resultado:** se guardado, 200 `image/jpeg`, crop 112×112, `Cache-Control:no-store`; request ID apenas no header. **Erros:** 401, 404 face/face_image_not_found.

### `DELETE /v1/collections/{collection_id}/persons/{person_id}/faces/{face_id}`

**Entrada:** três IDs. **Resultado:** elimina embedding/crop/linha do índice, 204. **Erros:** 404, 503.

## Pesquisa

### `POST /v1/collections/{collection_id}/search`

Com prova de vida em `normal`, uma falsificação ou uma consulta inadequada devolve HTTP 422 `liveness_fake` / `liveness_input_rejected` e `error.details.liveness`; a pesquisa não é executada. Isto difere de uma lista vazia de correspondências bem-sucedida. `observe` continua e devolve o resultado no rosto consultado.

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

Com prova de vida em `normal`, os rostos bloqueados têm `status: liveness_blocked` e um resultado separado. Contam em `liveness_blocked_faces`, não em `unknown_faces`, e não geram eventos de entrada. `observe` continua o reconhecimento. As entradas rejeitadas e as falsificações são apresentadas separadamente.

**Uso:** Consultar o estado atual em clientes sem interface. **Entrada:** ID do Monitor. **Resultado:** 200 com ligação, FPS efetivo, latência, saltos, rostos reconhecidos/desconhecidos, preview, reconexões e erro seguro, sem embeddings. **Erros:** 401, 404.

### `GET /v1/monitors/{monitor_id}/events`

**Uso:** Obter eventos voláteis de entrada/saída/erro/recuperação. **Entrada:** `limit` 1–1000 e último `cursor` opaco. **Resultado:** 200 com `events`, `next_cursor`, `truncated` e `stream_reset`; reiniciar perde eventos. **Erros:** 400 `invalid_cursor`, 401, 404.

### `GET /v1/monitors/{monitor_id}/preview.mjpeg`

**Uso:** Abrir o preview MJPEG bruto, desativado por predefinição. **Entrada:** ID e cabeçalho Bearer normal; nunca a chave no URL. **Resultado:** `multipart/x-mixed-replace` longo, codificado só com observadores; o cliente desenha caixas via `/state`. **Erros:** 401, 404, 409 `preview_disabled`, 503.

## Repetição de pedidos

GET pode ser repetido. Verifique o estado antes de repetir DELETE. Se o resultado de criar Person/Face for incerto pela rede, consulte o ID antes de novo POST. Repita apenas 429 e 503 transitórios com backoff exponencial limitado e jitter; corrija os 4xx.
