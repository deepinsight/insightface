# InsightFace Server

**Idiomas:** [English](README.md) · [中文](README.zh-CN.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [Français](README.fr.md) · [Русский](README.ru.md) · Português · [한국어](README.ko.md)

> **Uma GPU. 50M+ de vetores faciais. Pesquisa ultrarrápida com quantização INT8 de características e sem perda material de precisão.**

**Servidor de reconhecimento facial autoalojado com Web UI, REST API simples,
SQLite e inferência local por CPU ou GPU NVIDIA num único contentor.**

```text
carregar uma imagem -> detetar, verificar a vivacidade, comparar, registar ou pesquisar
```

> **Licença do modelo:** os modelos públicos pré-treinados InsightFace estão
> normalmente disponíveis apenas para investigação não comercial. O uso
> comercial exige autorização separada da
> [InsightFace](https://www.insightface.ai).

InsightFace Server é uma alternativa mais simples e centrada na privacidade ao
AWS Rekognition para fluxos comuns em infraestrutura própria. Imagens,
embeddings, modelos e índices podem permanecer na sua rede. Não é um substituto
compatível com AWS e não implementa SigV4, IAM, Region ou semântica de recursos
AWS.

Versão atual: **0.3.1**, Linux x86_64.

| Ambiente | Image |
| --- | --- |
| CPU | `ghcr.io/deepinsight/insightface-server:0.3.1-cpu` |
| GPU NVIDIA | `ghcr.io/deepinsight/insightface-server:0.3.1-cuda12` |

As tags móveis `cpu` e `cuda12` indicam a versão estável mais recente de cada
família. Não existe a tag ambígua `latest`. Consulte
[Maintainer Guide — English](docs/maintainer-guide.md) para a política.

**Novidades da 0.3.1 e atualização:** Server e o instalador executam como root; uma única montagem de modelos com escrita substitui a montagem separada de addons. O Compose cria o diretório quando necessário, sem configurar UID/GID ou permissões manualmente. Atualize também os ficheiros Compose e as suas sobreposições, preservando os caminhos de modelos, configuração e dados. Consulte os [passos de atualização](docs/user-guide.pt.md#atualizar-para-031).

Desde a 0.3.0 são suportados `raccoon_s`, `raccoon_l`, os seus manifestos, prova de vida opcional com instalação Web e imagens BMP. Desde essa versão, os resultados API e SDK omitem `model_version`. A prova de vida continua desativada até ser ativada.

![Dashboard InsightFace Server em inglês](docs/images/customer/dashboard-en.jpg)

## Principais funcionalidades

- Deteção SCRFD, cinco landmarks, alinhamento, embeddings ArcFace, L2
  normalization, cosine similarity original e pesquisa Person 1:N exata.
- Deteção multirresolução com um único NMS após fusão e seleção
  `largest` ou `center_largest`.
- `Collection -> Person -> FaceSample`, Collections ligadas ao modelo,
  registo multi-imagem com sucesso parcial, metadata e motivos explícitos.
- `review_mode` de registo: `off`, `standard` ou `strict`; embeddings
  pré-calculados opcionais por `external_trusted`.
- Pesquisa GPU exata com armazenamento vetorial FP32, FP16, BF16 e INT8.
- Web UI multilingue para Dashboard, Collections, People, Detect, Compare,
  Search, monitorização RTSP, System e Help.
- 31 operações REST snake_case em `/v1`, incluindo `/v1/embeddings` protegido,
  mais SDK Python leve e tipado.
- RTSP Monitors persistentes no servidor, eventos limitados em memória, vários
  clientes e `preview.mjpeg` opcional; fechar o browser não para a monitorização.
- SQLite como fonte persistente, índices exatos reconstruíveis em memória,
  uma única montagem de modelos com escrita e um diretório de configuração gravável, `/data` persistente, migrations, health checks e
  validação CUDA estrita sem fallback CPU silencioso.
- Entrada JPEG, PNG, WebP e BMP; originais não são guardados por predefinição.

A prova de vida está desativada por predefinição: `inference.addons = []` e `addons.auto_download = []` em `server/config/server.toml`. **Sistema → Detecção de vivacidade** descarrega e verifica o modelo, guardando a ativação para o próximo reinício manual; reutiliza uma cópia verificada. Não há descarga no arranque. Um modelo ativado ausente impede o arranque e apresenta instruções de instalação. Cada rosto avaliado devolve os campos principais `status`, `is_live` e `live_score`; só `input_rejected` acrescenta uma explicação para o utilizador em `reason`. O modo predefinido é `normal`; o registo ignora a prova por predefinição (`liveness_on_registration = false`). Consulte [configuração, permissões Web e atualização](docs/user-guide.pt.md#addon-opcional-de-prova-de-vida).

### Desempenho de pesquisa GPU na RTX 5090

Numa NVIDIA GeForce RTX 5090 (32.607 MiB), o índice flat exato CUDA nativo
armazenou até **58,9M vetores de imagem de 512 dimensões em INT8**.

| Tipo de dados GPU | Máximo de vetores de imagem | 10M Top-5 p50 | 10M QPS em série |
| --- | ---: | ---: | ---: |
| FP32 | 15,8M | 12,84 ms | 77,85 |
| FP16 | 30,7M | 6,83 ms | 146,32 |
| BF16 | 30,7M | 6,83 ms | 146,33 |
| INT8 | **58,9M** | **3,84 ms** | **260,81** |

INT8 atingiu 3,73 vezes a capacidade medida e 3,35 vezes o throughput Top-5 de
FP32 sobre 10M. São medições apenas de GPU na mesma RTX 5090 com Driver
580.105.08 e CUDA 12.9. A capacidade é o limite isolado do índice nativo sem
modelos ONNX carregados nem carga do Server. A velocidade usa exatamente 10M
vetores de imagem, varrimento exato Top-5 residente na GPU, uma pesquisa em
curso, 10 warm-ups e 100 medições. A pesquisa é exata dentro de cada
representação armazenada; a quantização pode alterar scores face a FP32.
Produção deve reservar VRAM para modelos, requests, concorrência, reconstrução
do índice e allocator.

### Precisão MR-ALL multirracial do ICCV21-MFR

Avaliámos os perfis de pesquisa nativos no conjunto de teste multirracial (MR)
do [ICCV21-MFR](../challenges/iccv21-mfr/) com o protocolo MR-ALL 1:1 de todos
os pares e FAR `1e-6`. Todos os perfis usam os mesmos embeddings `buffalo_l` de
512 dimensões, normalizados com L2 e extraídos uma única vez pela Server API;
apenas mudam as representações de armazenamento e de cálculo da pesquisa.

| Perfil de pesquisa | MR-ALL com FAR 1e-6 | Limiar cosine | Diferença para FP32 |
| --- | ---: | ---: | ---: |
| FP32 | 91,249107 % | 0,407787 | — |
| FP16 | 91,249197 % | 0,407787 | +0,000090 pontos percentuais |
| BF16 | 91,248502 % | 0,407787 | -0,000605 pontos percentuais |
| **INT8** | **91,248005 %** | **0,407739** | **-0,001102 pontos percentuais** |

**O INT8 não apresenta perda material de precisão neste benchmark:**
com a apresentação de duas casas decimais usada pelo challenge, FP32 e INT8
obtêm ambos **91,25 % de MR-ALL**; a diferença sem arredondamento é de apenas
0,0011 pontos percentuais. Mantêm-se também as vantagens acima de 3,73 vezes a
capacidade medida e 3,35 vezes o throughput Top-5 sobre 10M. Esta comparação
mede a precisão do armazenamento e da pesquisa vetorial, não a inferência de
modelos INT8.

![Gestão de Collections em inglês](docs/images/customer/collections-en.jpg)

![RTSP Monitor em inglês; endereço privado ocultado](docs/images/customer/monitoring-en.jpg)

## Início rápido

Requisitos:

- Linux x86_64 com Docker Engine e Docker Compose;
- para CUDA, GPU NVIDIA suportada, NVIDIA Driver e NVIDIA Container Toolkit.

O host não precisa de Python, OpenCV, ONNX Runtime, CUDA Toolkit ou cuDNN.
As Images públicas não incluem modelos, dados de clientes, API Keys ou
configuração de produção.

Num checkout completo do InsightFace, instale um modelo em `server/.models`:

Execute os comandos na raiz do repositório com `server/config/server.toml` presente. Server e o instalador de modelos executam como root (`0:0`). O Compose cria `server/.models` se não existir e monta-o uma única vez com escrita em `/models`. O subdiretório `addons` é criado ao descarregar um addon. Não são necessários exports UID/GID nem preparação manual de diretórios ou permissões. O arranque normal não descarrega modelos; a prova de vida está desativada por predefinição.

```bash
docker compose -f server/deploy/compose.cpu.yml pull
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models install buffalo_l --accept-license
```

Opcionalmente pode configurar a prova de vida ao instalar o modelo; substitua o comando de instalação por:

```bash
docker compose -f server/deploy/compose.cpu.yml \
  run --rm models install buffalo_l --accept-license --enable-liveness
```

Server não precisa de estar em execução. Numa instalação nova, o próximo `up -d` ativa a prova de vida; se Server já estiver em execução, use `docker compose -f server/deploy/compose.cpu.yml restart server`. `up -d` sozinho não recarrega as definições guardadas. Para CUDA use `compose.cuda12.yml`.

A ferramenta suporta os sete pacotes: `buffalo_l`, `buffalo_m`, `buffalo_s`,
`buffalo_sc`, `antelopev2`, `raccoon_s` e `raccoon_l`. Escreve
`manifest.json` e o `MODEL.LICENSE` assinado; `models verify` valida o pacote.
Os termos do modelo são separados da licença do código Server.

Iniciar CPU:

```bash
docker compose -f server/deploy/compose.cpu.yml up -d
curl -fsS http://127.0.0.1:18097/v1/health
```

Iniciar CUDA 12 em alternativa:

```bash
docker compose -f server/deploy/compose.cuda12.yml pull
docker compose -f server/deploy/compose.cuda12.yml \
  run --rm models install buffalo_l --accept-license
docker compose -f server/deploy/compose.cuda12.yml up -d
curl -fsS http://127.0.0.1:18098/v1/health
```

Abra `http://SERVIDOR:18097/` para CPU ou `http://SERVIDOR:18098/` para CUDA.
Crie uma Collection, registe uma Person com uma ou várias fotografias e
pesquise com outra. `docker compose ... down` sem `-v` preserva o volume.

Os Compose fornecidos desativam autenticação por predefinição para avaliação
isolada. Antes de expor o serviço a outros utilizadores ou redes:

```bash
export INSIGHTFACE_AUTH_ENABLED=true
export INSIGHTFACE_API_KEY='substitua-por-um-segredo-aleatorio-longo'
docker compose -f server/deploy/compose.cpu.yml up -d
```

Consulte o [guia para iniciantes](docs/user-guide.pt.md) para o fluxo completo.

## Compilar a partir do código

Pode compilar diretamente a partir de um diretório local com o código fonte
completo, incluindo alterações sem commit ou um diretório sem `.git`. Fazer
commits ou enviá-los com Git não é um requisito para compilar.

Os Dockerfiles copiam `server/` e módulos de inferência selecionados de
`python-package/insightface/`; por isso o diretório completo do código fonte é o contexto
de build.

CPU:

```bash
make -C server build-cpu
docker compose -f server/deploy/compose.cpu.yml \
  run --rm --pull never models install buffalo_l --accept-license
docker compose -f server/deploy/compose.cpu.yml \
  up -d --no-build --pull never
```

CUDA 12:

```bash
make -C server build-cuda12
docker compose -f server/deploy/compose.cuda12.yml \
  run --rm --pull never models install buffalo_l --accept-license
docker compose -f server/deploy/compose.cuda12.yml \
  up -d --no-build --pull never
```

`--pull never` garante o uso da Image local. O build ainda descarrega Images
base e dependências fixadas; a instalação descarrega separadamente o pacote de
modelo cuja licença foi aceite.

## Comportamento essencial

- Similarity é o coseno original, não probabilidade. Thresholds usam
  `0.0..1.0`, com predefinição `0.4`.
- Uma Collection fixa modelo e embedding contract. Em caso de divergência
  continua visível, mas registo/pesquisa devolve `collection_model_mismatch`.
- O Detection Profile de arranque é copiado para novas Collections; depois pode
  ser alterado independentemente para pedidos seguintes.
- O armazenamento opcional guarda um bounding-box JPEG crop redimensionado
  para 112x112, não o original nem a entrada alinhada de reconhecimento; vem
  desligado por predefinição.
- Commits SQLite são autoritativos. O índice sincroniza antes da resposta de
  registo/remoção e é reconstruído de SQLite após reinício.
- Respostas incluem `x-request-id`; listas usam cursor opacos e assinados.

Campos, defaults, ciclos de vida e erros exatos são mantidos apenas nos
documentos detalhados abaixo.

## API e SDK

Grupos principais:

- sistema: `/v1/health`, `/v1/system`, `/v1/models`;
- faces sem estado: `/v1/detect`, `/v1/compare`, `/v1/embeddings`;
- CRUD de Collection, Person e FaceSample;
- pesquisa Person numa Collection;
- configuração, estado, eventos e preview de RTSP Monitor.

O [guia REST API](docs/api.pt.md) contém todos os parâmetros, respostas, erros e
exemplos. OpenAPI interativo continua em `/docs`.

```python
from insightface_server import Client

with Client("http://localhost:18097", api_key=None) as client:
    faces = client.detect("photo.jpg")
    matches = client.search("employees", "unknown.jpg", limit=5)
```

A instalação, entradas, métodos e fluxos completos do SDK estão no
[guia do utilizador](docs/user-guide.pt.md).

## Segurança

A instalação fornecida usa root com as capabilities predefinidas do Docker e pode escrever nas montagens de modelos, configuração e dados. O restante sistema de ficheiros continua só de leitura e `no-new-privileges` mantém-se ativo. Limite o acesso ao host e os diretórios montados.

Imagens faciais e embeddings são dados biométricos. Em rede, ative autenticação,
termine HTTPS num reverse proxy de confiança, restrinja Docker e volumes,
mantenha CORS amplo desligado e defina backup, retenção, eliminação,
consentimento e resposta a incidentes. Não registe imagens, embeddings,
credenciais RTSP ou API Keys.

O Server não inclui TLS, contas, RBAC, cloud IAM ou camada jurídica. Operação e
segurança estão no [guia do utilizador](docs/user-guide.pt.md).

## Âmbito da primeira fase

Não inclui compatibilidade AWS/CompreFace, CUDA 11, Jetson, ARM64, Windows
Container, TensorRT, Kubernetes, Workers distribuídos, eventos Monitor
persistentes ou gravação/NVR, deepfake ou atributos demográficos.

## Documentação

- [Guia do utilizador](docs/user-guide.pt.md) — instalação, configuração,
  modelos, Web UI, SDK, GPU, segurança, backup e resolução de problemas.
- [Guia REST API](docs/api.pt.md) — todos os endpoints, campos, comportamentos,
  resultados, erros, paginação e exemplos.
- [Maintainer Guide — English](docs/maintainer-guide.md) — arquitetura,
  pesquisa interna, testes, contribuições e releases de contentores.

GitHub e a ajuda Web UI leem os mesmos Markdown localizados; só muda a
apresentação.

## Licença

O ponto único de licenciamento é [LICENSING.md](LICENSING.md). O código Server
e o SDK Python usam MIT License; esta declaração não cobre ficheiros ou pesos
de modelos, datasets ou componentes de terceiros. Modelos públicos InsightFace
estão normalmente limitados a investigação não comercial sem autorização
separada. Licenciamento comercial: <https://www.insightface.ai>.
