import { paramsType } from "../main";

export default function generateHeightmapHandler(params: paramsType, mesh: THREE.Mesh<THREE.PlaneGeometry, THREE.MeshBasicMaterial>, texture: THREE.Texture, loader: THREE.TextureLoader, mapsArray: Array<Array<string>>, heightmapsArray: Float64Array[], mapsContainer: HTMLElement, heightmap2d: HTMLImageElement, texture2d: HTMLImageElement, ctx: CanvasRenderingContext2D, worldWidth: number, worldDepth: number) {
    const generateHeightmap = () => {
        fetch('http://127.0.0.1:8000/generate', {
            method: 'POST',
            body: JSON.stringify({ 'dataset': params.currentDataset }),
            headers: {
                'Content-Type': 'application/json'
            },
        }).then(response => {
            console.log(response);
            response.json().then(res => {
                let heightmap: Float64Array = res.heightmap.split(',').map(Number);
                if (params.currentDataset === 'death_valley') { // -83 to 575
                    heightmap = heightmap.map(x => x * (575+83) - 83);
                } else if (params.currentDataset === 'mt_rainier') {
                    heightmap = heightmap.map(x => x * 2537 + 608);
                } else if (params.currentDataset === 'laytonville') {
                    heightmap = heightmap.map(x => x * 868 + 120);
                } else if (params.currentDataset === 'san_gabriel') {
                    heightmap = heightmap.map(x => x * 1394 + 484);
                } else if (params.currentDataset === 'post_earthquake') {
                    heightmap = heightmap.map(x => x * 237 + 665);
                } 
                heightmapsArray.push(heightmap);
                const imgSrc = `data:image/png;base64,${res.image64}`;
                texture = loader.load(imgSrc);
                mesh.material.map = texture;

                const img = new Image();
                img.className = 'heightmap_preview';
                const textureImg = new Image();
                textureImg.className = 'texture_preview';
                const inputSrc = imgSrc;
                img.src = inputSrc;
                textureImg.src = '';
                heightmap2d.src = inputSrc;
                texture = loader.load(inputSrc);

                const currentMaps = [inputSrc, ''];
                mapsArray.push(currentMaps);
                const divMapsContainer = document.createElement('div');
                divMapsContainer.appendChild(img);
                divMapsContainer.appendChild(textureImg);
                mapsContainer.appendChild(divMapsContainer);

                const thisMapId = mapsArray.length - 1;

                divMapsContainer.addEventListener('click', () => {
                    ctx.drawImage(img, 0, 0);
                    const pixels = ctx.getImageData(0, 0, worldWidth, worldDepth).data;
                    const vertices = mesh.geometry.attributes.position.array as Array<number>;
                    for (let i = 0; i < worldWidth * worldWidth; i++) {
                        vertices[1 + i * 3] = pixels[0 + i * 4] * 10
                    }
                    mesh.geometry.attributes.position.needsUpdate = true;

                    params.currentId = thisMapId;
                    heightmap2d.src = inputSrc;
                    texture = loader.load(inputSrc);
                    mesh.material.map = texture;
                    texture2d.src = mapsArray[thisMapId][1];

                    console.log(params.currentId, thisMapId, mapsArray);
                    console.log(mapsArray[thisMapId][1])
                })

            })
        })
    }
    const buttonGenerateHeightmap = document.getElementById('btn_generate_heightmap') as HTMLElement;
    buttonGenerateHeightmap.addEventListener('click', generateHeightmap);
}