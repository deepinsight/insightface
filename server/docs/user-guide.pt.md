# Guia do utilizador do InsightFace Server

**Idiomas:** [English](user-guide.md) · [中文](user-guide.zh-CN.md) · [日本語](user-guide.ja.md) · [Deutsch](user-guide.de.md) · [Español](user-guide.es.md) · [Français](user-guide.fr.md) · [Русский](user-guide.ru.md) · Português · [한국어](user-guide.ko.md)

Este guia conduz um novo utilizador desde uma pasta vazia até à primeira pesquisa bem-sucedida. As mesmas funções estão disponíveis na Web UI, em `/v1` e no SDK Python. Consulte todos os campos e resultados HTTP no [guia da API](api.pt.md).

Os modelos são identificados por `model_id`; as respostas não incluem um `model_version` separado.

Atualizar o Server com o mesmo modelo de reconhecimento e contrato preserva `embedding_contract_id`, amostras e embeddings das Collections existentes. Trocar de modelo é uma migração separada; um contrato incompatível causa `collection_model_mismatch` no registo e na pesquisa.

Para usar a prova de vida, consulte [configuração, instalação e resultados](#addon-opcional-de-prova-de-vida). Cada operação explica também os seus efeitos.

## Do zero à primeira pesquisa

CPU requer Linux x86_64, Docker Engine e Docker Compose. CUDA requer ainda um Driver NVIDIA compatível e NVIDIA Container Toolkit; não instale CUDA, cuDNN, ORT, Python ou OpenCV no anfitrião.

Execute os comandos na raiz do repositório com `server/config/server.toml` presente. Server e o instalador de modelos executam como root (`0:0`). O Compose cria `server/.models` se não existir e monta-o uma única vez com escrita em `/models`. O subdiretório `addons` é criado ao descarregar um addon. Não são necessários exports UID/GID nem preparação manual de diretórios ou permissões. O arranque normal não descarrega modelos; a prova de vida está desativada por predefinição.

```bash
docker compose -f server/deploy/compose.cpu.yml pull
docker compose -f server/deploy/compose.cpu.yml run --rm models install buffalo_l
docker compose -f server/deploy/compose.cpu.yml up -d
curl -fsS http://127.0.0.1:18097/v1/health
```

Opcionalmente pode configurar a prova de vida ao instalar o modelo; substitua o comando de instalação por:

```bash
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models install buffalo_l --accept-license --enable-liveness
```

Server não precisa de estar em execução. Numa instalação nova, o próximo `up -d` ativa a prova de vida; se Server já estiver em execução, use `docker compose -f server/deploy/compose.cpu.yml restart server`. `up -d` sozinho não recarrega as definições guardadas. Para CUDA use `compose.cuda12.yml`.

Para GPU use `compose.cuda12.yml` e a porta `18098`. O instalador mostra a licença antes do download; os modelos públicos InsightFace destinam-se apenas a investigação não comercial sem uma licença comercial separada.

O Compose fornecido desativa a autenticação por predefinição para avaliação isolada. Antes de expor o serviço, defina `INSIGHTFACE_AUTH_ENABLED=true` e uma `INSIGHTFACE_API_KEY` longa. Verifique depois o Dashboard, crie uma Collection, registe uma Person e pesquise com outra imagem. Pare com `docker compose ... down` sem `-v` para preservar o volume.

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

Em **Sistema → Detecção de vivacidade**, selecione **Transferir e ativar após reiniciar**. Após verificar SHA-256, acrescenta-se `liveness` às duas listas, preservando as outras entradas, comentários e opções. Um ficheiro verificado é reutilizado. **Reinicie manualmente o Server** para aplicar. Os erros permitem repetir; uma descarga falhada não ativa a prova de vida.

Sistema distingue instalação verificada (`installed`), execução atual (`enabled`), configuração guardada para o próximo arranque (`configured_enabled`) e necessidade de reinício (`restart_required`). Descarregar ou guardar não altera a inferência em curso. Para desativar, guarde `inference.addons=[]` e `addons.auto_download=[]` no mesmo ficheiro e reinicie manualmente. A ação Web não altera a definição de registo; o valor predefinido continua a ser `liveness_on_registration=false`.

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

```bash
docker compose -f server/deploy/compose.cpu.yml run --rm models addons install liveness
docker compose -f server/deploy/compose.cpu.yml run --rm models addons verify liveness
docker compose -f server/deploy/compose.cpu.yml up -d --force-recreate
```

Um modelo ativado ausente impede o arranque com `addon_model_missing`; um inválido produz `addon_model_invalid`. O addon não é desativado silenciosamente.

Atualizar apenas o código com a prova de vida desativada mantém as Collections
e os embeddings existentes utilizáveis. As migrações da base de dados preservam
as amostras e os embeddings; a prova de vida não altera o resumo criptográfico
do modelo de reconhecimento nem o `embedding_contract_id`.

### Montagens e permissões para descargas Web

Os ficheiros Compose fornecidos montam todo `/models` com escrita; não é necessária uma montagem separada de `/models/addons`. Server e o instalador usam root (`0:0`), e a instalação cria `addons` quando necessário. Todo o diretório existente `server/config` é montado com escrita em `/etc/insightface` nos dois serviços para que a ação Web e `--enable-liveness` guardem `server.toml` de forma atómica. Esta configuração não exige exports UID/GID, `chgrp` ou `chmod`. Ficheiros ou montagens deliberadamente só de leitura continuam a impedir a gestão Web; confirme as permissões das montagens personalizadas.

A montagem de modelos de cada serviço usa `create_host_path: true`. Os serviços mantêm as capabilities predefinidas do Docker; não é aplicado `cap_drop: [ALL]`. O restante sistema de ficheiros continua só de leitura e `no-new-privileges` mantém-se ativo. Root pode alterar ficheiros nas montagens com escrita; as novas descargas podem pertencer a root no host.

Em instalações personalizadas use os caminhos reais; para CUDA use `compose.cuda12.yml`. As montagens antigas só de leitura continuam a funcionar com a prova de vida desativada. A ação Web explica a indisponibilidade; pode também instalar pela CLI e editar a configuração manualmente. Depois de guardar na Web, aplique com `docker compose -f server/deploy/compose.cpu.yml restart server`. Alterar montagens ou variáveis de proxy exige recriar o contentor.

Se precisar de proxy, defina `HTTP_PROXY`, `HTTPS_PROXY` e `NO_PROXY` antes de criar o contentor; o Compose passa-as ao Server e à ferramenta de modelos. Use um endereço LAN acessível a partir do contentor: o seu `127.0.0.1` não é o Mac. A ação usa a autenticação API Key existente; sem autenticação, qualquer cliente com acesso à API também a pode executar. Só descarrega o modelo de prova de vida publicado e fixado, sem URLs arbitrários nem troca do pacote de modelo base.

### Resultados da prova de vida

| Resultado | `status` | `is_live` | `live_score` |
| --- | --- | --- | --- |
| Prova aprovada | `ok` | `true` | `[0, 1]` |
| Não vivo | `ok` | `false` | `[0, 1]` |
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

`normal` reconhece apenas rostos aprovados; `observe` registra o resultado e continua o reconhecimento. Sem avaliação, `liveness` é omitido. Os três campos principais são `status`, `is_live` e `live_score`: aprovado/fake usa `status: ok`, booleano e pontuação; entrada rejeitada usa `status: input_rejected` e dois valores `null`.

Detect retorna HTTP 200 mesmo para resultados negativos. Em `normal`, embeddings, comparação e busca retornam HTTP 422 `liveness_fake` ou `liveness_input_rejected` com `error.details.liveness`; comparação acrescenta `details.side`. Falhas retornam HTTP 503 `liveness_unavailable`. As falhas de execução interrompem a operação tanto em `normal` como em `observe`; não são convertidas em `input_rejected`.

O cadastro de pessoas e a adição de FaceSamples ignoram a prova de vida por padrão: `[inference].liveness_on_registration=false` não executa o modelo e omite `liveness` nas novas amostras. Com `true` e o addon habilitado, aplica-se `normal`/`observe`; rejeições incluem `reason` e `liveness`. A revisão de qualidade por `review_mode` e a validação dos embeddings externos continuam ativas. `review_mode=off` e `external_trusted` não ignoram uma prova de cadastro habilitada. As requisições não podem alterar esta configuração de inicialização. Os resultados já salvos continuam disponíveis.

RTSP distingue `liveness_blocked` de `unknown` e conta `liveness_blocked_faces`. Rostos bloqueados não geram eventos de entrada de pessoas/desconhecidos e reiniciam a confirmação. Falhas de inferência apagam identidades exibidas anteriormente.

`liveness_compare_scope` escolhe `both` (predefinição), `source` ou `target` para `/v1/compare`. A prova passa quando `live_score >= liveness_threshold`.

O modelo é guardado em `server/.models/addons/liveness.onnx` no anfitrião e `/models/addons/liveness.onnx` no contentor. `addons` em `/v1/models` e `/v1/system` apresenta os addons ativos.

[Contrato completo da API](api.pt.md#addon-opcional-de-prova-de-vida).

## 1. Entrada e estado

Abra `http://SERVIDOR:18097/` para CPU ou `http://SERVIDOR:18098/` para CUDA 12. Se a autenticação estiver ativa, escolha **Configurar chave API**, introduza a chave do operador e use-a neste separador. Fica apenas em memória e desaparece ao atualizar ou fechar.

Em **Painel** ou **Sistema**, confirme que serviço, base de dados, modelos e Provider estão prontos. CUDA deve indicar `CUDAExecutionProvider` e nunca recua silenciosamente para CPU.

O Dashboard mostra sempre a prova de vida ativada ou desativada abaixo do modelo. Sistema distingue instalação, estado atual e reinício pendente.

## 2. Criar uma Collection

Em **Coleções** → **Nova coleção**, defina ID estável, nome, limiar cosine
(`0.4` inicialmente), perfil disponível, capacidade e máximo de FaceSamples por
pessoa. Guardar em JPEG um `bounding-box crop` redimensionado para 112×112 está
desligado por predefinição; não é a entrada alinhada de reconhecimento.

A Collection fica ligada ao ID, digest, dimensão e pré-processamento do modelo. Após mudar o modelo, continua visível, mas registo e pesquisa são recusados quando o contrato não coincide.

O perfil de deteção copia os valores do sistema ao criar a Collection e depois permite alterar tamanhos de entrada, limiares de deteção/NMS e estratégia de um rosto. `largest` prioriza a área; `center_largest` maximiza `área - 2,0 × distância em píxeis ao quadrado entre o centro da caixa e o da imagem`. A confiança de deteção não participa nesta pontuação.

## 3. Registar uma Person

Em **Pessoas**, selecione a Collection e **Registar pessoa**. Pode indicar ID, nome, ID externo, metadata JSON e várias imagens JPEG, PNG, WebP ou BMP.

- `off`: usa a estratégia de um rosto da Collection e permite vários rostos;
- `standard`: exige um rosto utilizável e verifica tamanho, deteção, nitidez, brilho e pose;
- `strict`: exige também que a melhor similaridade interna seja superior à melhor similaridade com outra pessoa.

O lote aceita sucesso parcial e explica cada rejeição. Os originais não são guardados. `external_trusted` aceita um embedding normalizado L2; a imagem continua obrigatória para deteção e qualidade, mas o vetor não é extraído novamente.

Criar Person e adicionar FaceSamples ignora a prova de vida por predefinição (`liveness_on_registration=false`). Se ativada, `normal` rejeita fake/entrada inadequada; `observe` guarda o resultado e continua. A revisão de qualidade segue o `review_mode` selecionado. A lista de rejeições mostra o `reason` real e a prova de vida separadamente.

## 4. Detetar, comparar e pesquisar

**Detetar** mostra caixas, cinco pontos, pontuação e qualidade; sem rostos devolve uma lista vazia válida. **Comparar** usa o perfil do sistema ou da Collection para escolher um rosto por imagem e devolve `similarity` cosine, `threshold` e `matched`. Similaridade não é probabilidade.

Em **Pesquisar**, escolha Collection e imagem. A pontuação da pessoa é a maior similaridade dos seus FaceSamples. Os resultados são decrescentes; sem correspondência é uma lista vazia. Cada amostra é confirmada no SQLite e adicionada ao índice antes da resposta. No reinício, o índice é reconstruído a partir do SQLite.

Cada rosto avaliado inclui `liveness.status`, `liveness.is_live` e `liveness.live_score`. Fake e `input_rejected` também devolvem HTTP 200, sem extrair características de reconhecimento. `input_rejected` indica uma área de imagem insuficiente em redor do rosto; `liveness.reason` explica como ajustar a imagem. Sem `liveness`, não houve avaliação.

`liveness_compare_scope` (`both`, `source`, `target`) escolhe os lados avaliados antes do reconhecimento. Em `normal`, a rejeição devolve HTTP 422 `liveness_fake` / `liveness_input_rejected`, `error.details.liveness` e `error.details.side`, sem similaridade. `observe` continua e inclui os resultados nos rostos avaliados.

Com prova de vida em `normal`, fake/consulta inadequada devolve HTTP 422 `liveness_fake` / `liveness_input_rejected` e `error.details.liveness`; a pesquisa não é executada. Isto difere de uma lista vazia de correspondências bem-sucedida. `observe` continua e devolve o resultado no rosto consultado.

## 5. Monitorização de câmara RTSP

Em **Monitorização de câmaras**, crie um Monitor persistente e configure origem RTSP, Collection, frequência, limiar opcional e política de eventos. A pré-visualização está desligada por predefinição; reconhecimento e eventos continuam sem ela. Quando ativa, a Web UI desenha pessoas registadas a verde e desconhecidas a laranja a partir de `/state` sobre imagens brutas.

O Monitor funciona independentemente do navegador e tarefas ativas são restauradas ao reiniciar o servidor. A configuração fica no SQLite e credenciais RTSP encriptadas em `/data`, mas fotogramas e eventos não são guardados. Eventos ficam apenas num buffer de memória limitado. O descodificador mantém o último fotograma e ignora os antigos em vez de os acumular.

Com prova de vida em `normal`, os rostos bloqueados têm `status: liveness_blocked` e um resultado separado. Contam em `liveness_blocked_faces`, não em `unknown_faces`, e não geram eventos de entrada. `observe` continua o reconhecimento. Entrada rejeitada e fake são apresentados separadamente.

## 6. Dados e segurança

Mantenha e faça cópias de segurança de `/data`, do diretório de modelos e da configuração. A instalação fornecida usa root com as capabilities predefinidas do Docker e `/models` com escrita; o serviço pode alterar os modelos, a configuração e os dados montados. O restante sistema de ficheiros do contentor continua só de leitura e `no-new-privileges` mantém-se ativo; não é necessário `privileged`. Restrinja o acesso ao Docker e ao host, e não monte diretórios do host alheios ao serviço. Antes de operações em massa, copie SQLite e face crops juntos. As chaves são guardadas como hash; iniciar o mesmo volume com outro `INSIGHTFACE_API_KEY` roda a chave ativa. Não registe imagens, embeddings nem chaves.

O explorador de esquema OpenAPI para programadores está em `/docs`; as instruções práticas da API estão nesta ajuda. Inclua `x-request-id` ao comunicar problemas. `401` indica chave, `409 collection_model_mismatch` contrato do modelo e `422 face_not_found` ausência de rosto utilizável.

## 7. Modelos e licenças

As imagens não incluem modelos. O arranque normal fica offline; o serviço
pontual `models` instala em `server/.models`:

```bash
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models install buffalo_l --accept-license
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models verify buffalo_l
```

São suportados `buffalo_l` (`det_10g.onnx` + `w600k_r50.onnx`), `buffalo_m`,
`buffalo_s`, `buffalo_sc`, `antelopev2`, `raccoon_s` e `raccoon_l`. A
instalação cria `manifest.json` e
`MODEL.LICENSE` assinada. Sem `--accept-license`, um terminal interativo pede
confirmação antes de descarregar; em execução não interativa, o parâmetro é
obrigatório e, sem ele, o comando termina sem descarregar. Os modelos públicos
pré-treinados do InsightFace são apenas para investigação não comercial sem licença comercial separada.

`raccoon_s` e `raccoon_l` são suportados. O Server instala apenas deteção e reconhecimento de cada pacote; não carrega o verificador Raccoon. O nome identifica o modelo, sem número de versão separado. A ação Web de prova de vida não troca o modelo base. Para outro modelo de reconhecimento, use uma Collection compatível; os embeddings anteriores não passam a representar características do novo modelo.

## 8. Configuração de arranque e pesquisa

```toml
[inference]
addons = []
liveness_mode = "normal"
liveness_threshold = 0.8
liveness_compare_scope = "both"
liveness_on_registration = false

[addons]
auto_download = []
```

`server/config/server.toml` é lido uma vez ao iniciar; alterações requerem
reinício. Predefinições: `input_sizes=[[96,96],[512,512]]`, limiar de deteção
`0.50`, NMS `0.40`, `single_face_selection="largest"` e máximo 100 rostos.
SCRFD executa cada resolução, converte candidatos para a imagem original e faz
um único NMS global. `max_concurrency="auto"` significa CPU 4 e CUDA 8.
`[web].disabled=true` mantém apenas `/v1` e `/openapi.json`.

System anuncia somente perfis disponíveis. O perfil fica fixo quando se cria a
Collection e não pode mudar por pedido:

- `fp32_v1`: CPU/CUDA padrão;
- `fp16_v1`: CUDA;
- `bf16_v1`: CPU compatível ou CUDA SM80+;
- `int8_x736_v1`: INT8 recomendado CPU/CUDA, acumulação INT32;
- `int8_x1000_v1`: compatibilidade com Collections existentes.

Todos percorrem cada FaceSample, não são índices ANN e devolvem raw cosine.
`capacity_rows=100000`, limite global `10000000` e
`max_faces_per_person=20`. Em 512 dimensões, só o vetor ocupa aproximadamente
2 048 bytes FP32, 1 024 FP16/BF16 ou 512 INT8 por linha.

## 9. SDK, compilação e operação de dados

O SDK Python aceita caminho, bytes e file-like object e fornece métodos tipados
para Detect, Compare, Collections, registo, Search e Monitors. Consulte o
contrato HTTP no [guia da API](api.pt.md).

Pode compilar diretamente a partir de um diretório local com o código fonte
completo, incluindo alterações sem commit ou um diretório sem `.git`. Fazer
commits ou enviá-los com Git não é um requisito para compilar.

```bash
make -C server build-cpu
make -C server build-cuda12
```

Depois de os testes passarem, publique a mesma imagem que foi testada. Fazer
mais tarde commit do mesmo código ou organizar os commits não exige nova
compilação. Alterações aos ficheiros incluídos na imagem, como código, recursos
do frontend ou ajuda do utilizador integrada, exigem nova compilação e
validação.

Use `--pull never` no Compose para a imagem local. Os tags imutáveis são
`0.3.1-cpu` e `0.3.1-cuda12`; `cpu` e `cuda12` apontam à última versão estável,
sem tag `latest`. Antes de atualizar, pare escritas e faça backup SQLite-safe de
`/data` e crops. Não use `docker compose down -v`, pois elimina o volume.

### Atualizar para 0.3.1

A 0.3.1 simplifica a instalação: Server e o instalador usam root, o Compose cria o diretório de modelos quando necessário e um único `/models` com escrita substitui a montagem separada de addons.

Desde a 0.3.0 são suportados `raccoon_s`, `raccoon_l`, os seus manifestos, prova de vida opcional, instalação Web de addons e imagens BMP. Server usa a deteção e o reconhecimento do Raccoon; o verificador não é carregado. Estas funcionalidades e os contratos de resposta API mantêm-se inalterados na 0.3.1.

**1.** Atualize o código do Server e os ficheiros Compose para a versão 0.3.1,
mantendo as definições de `server/config/server.toml` e as personalizações
da instalação. Preserve o caminho dos modelos, o nome do volume `/data`,
o armazenamento de crops, as portas e as definições da API Key. Em ficheiros
Compose personalizados, atualize as imagens dos serviços `server` e `models`
para `0.3.1-cpu` ou `0.3.1-cuda12`, conforme o ambiente. Aplique aos comandos
seguintes os mesmos ficheiros Compose, sobreposições de configuração e nome
de projeto que usa normalmente.

Antes de iniciar, atualize também as suas sobreposições Compose, não apenas os tags: ambos os serviços devem usar `user: "0:0"`, sem `cap_drop: [ALL]` nem os antigos ajustes UID/GID ou `group_add`. Use em ambos uma única montagem bind com escrita em `/models` e `create_host_path: true`; remova a montagem separada de `/models/addons` e `x-addons-path`. Preserve o caminho real dos modelos e os ficheiros existentes, incluindo addons. As montagens de configuração mantêm `create_host_path: false`, por isso `server/config` e `server.toml` têm de continuar presentes. Preserve o volume de dados e faça uma cópia de dados, modelos e configuração antes da alteração. Remova os antigos valores de utilizador e montagens das suas sobreposições; alterar só a imagem não aplica estas mudanças.

Mantenha o sistema de ficheiros do contentor só de leitura e `no-new-privileges`. Os dois serviços precisam de todo o diretório de configuração com escrita. Substitua a antiga montagem do ficheiro único só de leitura do instalador pela montagem do diretório. A instalação padrão não exige alterar recursivamente as permissões existentes nem preparar um diretório de addons.

**2.** Descarregue as novas imagens e recrie o contentor Server. Na raiz do
repositório, escolha os comandos da instalação existente:

CPU:

```bash
docker compose -f server/deploy/compose.cpu.yml pull server models
docker compose -f server/deploy/compose.cpu.yml up -d --no-build --force-recreate server
curl -fsS http://127.0.0.1:18097/v1/health
```

CUDA:

```bash
docker compose -f server/deploy/compose.cuda12.yml pull server models
docker compose -f server/deploy/compose.cuda12.yml up -d --no-build --force-recreate server
curl -fsS http://127.0.0.1:18098/v1/health
```

Se compilar localmente, compile primeiro as imagens 0.3.1 e use
`up -d --no-build --pull never --force-recreate server` em vez de descarregar.
`docker compose restart` por si só não muda para uma nova imagem nem aplica
alterações às montagens.

**3.** O arranque aplica automaticamente as migrações da base de dados. Aguarde
que `/v1/health` indique `ready` e a versão `0.3.1`, depois confirme em
**Sistema** o modelo e o fornecedor de execução esperados. Verifique se as
Collections e pessoas existentes estão presentes e experimente uma pesquisa
com resultado conhecido. Manter o mesmo modelo e contrato de embeddings
preserva as amostras, os embeddings e os identificadores de contrato das
Collections; não é necessário voltar a registar as pessoas.

**A prova de vida é opcional após a atualização.** A configuração fornecida e
as configurações antigas sem chaves de addons mantêm-na desativada; atualizar
o Server, por si só, não exige descarregar o modelo de prova de vida. O arranque
do Server nunca descarrega modelos. Para a ativar, siga a
[configuração da prova de vida](#addon-opcional-de-prova-de-vida): prepare as
[montagens e permissões para descargas Web](#montagens-e-permissões-para-descargas-web),
escolha **Sistema → Detecção de vivacidade → Transferir e ativar após reiniciar**,
aguarde a instalação e a gravação da configuração com sucesso e reinicie
manualmente o Server. As predefinições são `normal`, limiar `0.8` e
`liveness_on_registration=false`. O modelo permanece em
`<models_dir>/addons/liveness.onnx`.

**Adotar Raccoon é uma mudança de modelo separada.** Atualizar o Server mantém
o pacote de modelos atual. Para adotar `raccoon_s` ou `raccoon_l`, instale o
pacote escolhido num diretório de modelos separado seguindo as
[instruções de instalação](#7-modelos-e-licenças) e configure uma instalação
para o usar. As Collections têm de corresponder ao contrato de embeddings do
novo modelo; crie Collections compatíveis e volte a registar as pessoas ou
faça uma migração de dados separada. A Web UI não muda os pacotes de modelos
base.

**Compatibilidade da API e do SDK desde a 0.3.0:** Os resultados de modelos, Collections e
FaceSamples deixam de incluir `model_version`; a identidade do modelo usa
`model_id` e a compatibilidade da Collection usa `embedding_contract_id`.
Atualize os clientes que exigem o campo removido e use o SDK `0.3.1` ao
atualizar o cliente Python fornecido. Quando a prova de vida é avaliada,
`liveness` contém os campos principais `status`, `is_live` e `live_score`,
com `reason` apenas para `input_rejected`; quando não é
avaliada, o campo é omitido. Consulte as
[regras de resultados e erros](#resultados-da-prova-de-vida) antes de a ativar
nos pedidos de reconhecimento.

## 10. GPU, rede e resolução de problemas

A imagem CUDA contém CUDA Runtime 12.9.1, cuDNN 9.24.0 e
`onnxruntime-gpu==1.27.0`. Turing/Ampere/Ada/Hopper exigem R535+, Blackwell e
RTX 50 exigem 570.26+; recomenda-se R580 estável ou superior. No arranque são
verificados GPU, Compute Capability, Driver, CUDA/cuDNN/ORT, Provider, Sessions
reais e warm-up; fallback silencioso para CPU é recusado.

Ao expor a rede, termine HTTPS num reverse proxy de confiança, limite origins
CORS, rate, body e timeout, e proteja `/data` e backups como dados biométricos.
Não registe imagens, embeddings ou chaves. A fase um tem uma única API Key sem
roles e não é autorização multi-tenant.
