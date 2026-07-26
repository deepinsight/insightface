# Guia do utilizador do InsightFace Server

**Idiomas:** [English](user-guide.md) · [中文](user-guide.zh-CN.md) · [日本語](user-guide.ja.md) · [Deutsch](user-guide.de.md) · [Español](user-guide.es.md) · [Français](user-guide.fr.md) · [Русский](user-guide.ru.md) · Português · [한국어](user-guide.ko.md)

Este guia conduz um novo utilizador desde uma pasta vazia até à primeira pesquisa bem-sucedida. As mesmas funções estão disponíveis na Web UI, em `/v1` e no SDK Python. Consulte todos os campos e resultados HTTP no [guia da API](api.pt.md).

## Do zero à primeira pesquisa

CPU requer Linux x86_64, Docker Engine e Docker Compose. CUDA requer ainda um Driver NVIDIA compatível e NVIDIA Container Toolkit; não instale CUDA, cuDNN, ORT, Python ou OpenCV no anfitrião.

```bash
mkdir -p server/.models
docker compose -f server/deploy/compose.cpu.yml pull
docker compose -f server/deploy/compose.cpu.yml run --rm models install buffalo_l
docker compose -f server/deploy/compose.cpu.yml up -d
curl -fsS http://127.0.0.1:18097/v1/health
```

Para GPU use `compose.cuda12.yml` e a porta `18098`. O instalador mostra a licença antes do download; os modelos públicos InsightFace destinam-se apenas a investigação não comercial sem uma licença comercial separada.

O Compose fornecido desativa a autenticação por predefinição para avaliação isolada. Antes de expor o serviço, defina `INSIGHTFACE_AUTH_ENABLED=true` e uma `INSIGHTFACE_API_KEY` longa. Verifique depois o Dashboard, crie uma Collection, registe uma Person e pesquise com outra imagem. Pare com `docker compose ... down` sem `-v` para preservar o volume.

## 1. Entrada e estado

Abra `http://SERVIDOR:18097/` para CPU ou `http://SERVIDOR:18098/` para CUDA 12. Se a autenticação estiver ativa, escolha **Configurar chave API**, introduza a chave do operador e use-a neste separador. Fica apenas em memória e desaparece ao atualizar ou fechar.

Em **Painel** ou **Sistema**, confirme que serviço, base de dados, modelos e Provider estão prontos. CUDA deve indicar `CUDAExecutionProvider` e nunca recua silenciosamente para CPU.

## 2. Criar uma Collection

Em **Coleções** → **Nova coleção**, defina ID estável, nome, limiar cosine
(`0.4` inicialmente), perfil disponível, capacidade e máximo de FaceSamples por
pessoa. Guardar em JPEG um `bounding-box crop` redimensionado para 112×112 está
desligado por predefinição; não é a entrada alinhada de reconhecimento.

A Collection fica ligada ao ID, versão, digest, dimensão e pré-processamento do modelo. Após mudar o modelo, continua visível, mas registo e pesquisa são recusados quando o contrato não coincide.

O perfil de deteção copia os valores do sistema ao criar a Collection e depois permite alterar tamanhos de entrada, limiares de deteção/NMS e estratégia de um rosto. `largest` prioriza a área; `center_largest` maximiza `área - 2,0 × distância em píxeis ao quadrado entre o centro da caixa e o da imagem`. A confiança de deteção não participa nesta pontuação.

## 3. Registar uma Person

Em **Pessoas**, selecione a Collection e **Registar pessoa**. Pode indicar ID, nome, ID externo, metadata JSON e várias imagens JPEG, PNG ou WebP.

- `off`: usa a estratégia de um rosto da Collection e permite vários rostos;
- `standard`: exige um rosto utilizável e verifica tamanho, deteção, nitidez, brilho e pose;
- `strict`: exige também que a melhor similaridade interna seja superior à melhor similaridade com outra pessoa.

O lote aceita sucesso parcial e explica cada rejeição. Os originais não são guardados. `external_trusted` aceita um embedding normalizado L2; a imagem continua obrigatória para deteção e qualidade, mas o vetor não é extraído novamente.

## 4. Detetar, comparar e pesquisar

**Detetar** mostra caixas, cinco pontos, pontuação e qualidade; sem rostos devolve uma lista vazia válida. **Comparar** usa o perfil do sistema ou da Collection para escolher um rosto por imagem e devolve `similarity` cosine, `threshold` e `matched`. Similaridade não é probabilidade.

Em **Pesquisar**, escolha Collection e imagem. A pontuação da pessoa é a maior similaridade dos seus FaceSamples. Os resultados são decrescentes; sem correspondência é uma lista vazia. Cada amostra é confirmada no SQLite e adicionada ao índice antes da resposta. No reinício, o índice é reconstruído a partir do SQLite.

## 5. Monitorização de câmara RTSP

Em **Monitorização de câmaras**, crie um Monitor persistente e configure origem RTSP, Collection, frequência, limiar opcional e política de eventos. A pré-visualização está desligada por predefinição; reconhecimento e eventos continuam sem ela. Quando ativa, a Web UI desenha pessoas registadas a verde e desconhecidas a laranja a partir de `/state` sobre imagens brutas.

O Monitor funciona independentemente do navegador e tarefas ativas são restauradas ao reiniciar o servidor. A configuração fica no SQLite e credenciais RTSP encriptadas em `/data`, mas fotogramas e eventos não são guardados. Eventos ficam apenas num buffer de memória limitado. O descodificador mantém o último fotograma e ignora os antigos em vez de os acumular.

## 6. Dados e segurança

Mantenha `/data` persistente e `/models` só de leitura. Antes de operações em massa, copie SQLite e face crops juntos. As chaves são guardadas como hash; iniciar o mesmo volume com outro `INSIGHTFACE_API_KEY` roda a chave ativa. Não registe imagens, embeddings nem chaves.

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
`buffalo_sc` e `antelopev2`. A instalação cria `manifest.json` e
`MODEL.LICENSE` assinada. Sem `--accept-license`, a ferramenta mostra os termos
e termina sem descarregar. Os modelos públicos pré-treinados do InsightFace são
apenas para investigação não comercial sem licença comercial separada.

## 8. Configuração de arranque e pesquisa

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

Qualquer utilizador pode compilar a partir do repositório completo:

```bash
make -C server build-cpu
make -C server build-cuda12
```

Use `--pull never` no Compose para a imagem local. Os tags imutáveis são
`0.2.0-cpu` e `0.2.0-cuda12`; `cpu` e `cuda12` apontam à última versão estável,
sem tag `latest`. Antes de atualizar, pare escritas e faça backup SQLite-safe de
`/data` e crops. Não use `docker compose down -v`, pois elimina o volume.

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
