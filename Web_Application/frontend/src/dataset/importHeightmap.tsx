// @ts-ignore
import npyjs from "npyjs";

export default function importHeightmapHandler(params: any, mesh: THREE.Mesh<THREE.PlaneGeometry, THREE.MeshBasicMaterial>, texture: THREE.Texture, loader: THREE.TextureLoader, mapsArray: Array<Array<string>>, heightmapsArray: Float64Array[], mapsContainer: HTMLElement, heightmap2d: HTMLImageElement, texture2d: HTMLImageElement, ctx: CanvasRenderingContext2D, worldWidth: number, worldDepth: number) {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/png,image/jpeg,.npy';
    input.addEventListener('change', async () => {
        if (input.files && input.files[0]) {
            const read = new FileReader();
            read.readAsDataURL(input.files[0]);
            read.onloadend = () => {
                const img = new Image();
                img.className = 'heightmap_preview';
                const textureImg = new Image();
                textureImg.className = 'texture_preview';
                const newCanv = document.createElement('canvas');
                const newCtx = newCanv.getContext('2d') as CanvasRenderingContext2D;
                let n = new npyjs();
                // @ts-ignore
                n.load(read.result, (array) => {
                    console.log(
                        `You loaded an array with ${array.length} elements and ${array.shape} dimensions.`
                    );
                    const heightArray = array.data;
                    console.log(heightArray[0], heightArray[1])
                    const min = Math.min(...heightArray);
                    const max = Math.max(...heightArray);
                    const range = max - min;
                    for (let i = 0; i < heightArray.length; i++) {
                        heightArray[i] = (heightArray[i] - min) / range * 255;
                    }

                    // draw array to canvas
                    newCanv.width = array.shape[0];
                    newCanv.height = array.shape[1];
                    const imgData = newCtx.createImageData(array.shape[0], array.shape[1]);
                    for (let i = 0; i < heightArray.length; i++) {
                        imgData.data[i * 4] = heightArray[i];
                        imgData.data[i * 4 + 1] = heightArray[i];
                        imgData.data[i * 4 + 2] = heightArray[i];
                        imgData.data[i * 4 + 3] = 255;
                    }
                    newCtx.putImageData(imgData, 0, 0);
                    const inputSrc = newCanv.toDataURL();
                    img.src = inputSrc;
                    textureImg.src = '';
                    heightmap2d.src = inputSrc;
                    texture = loader.load(inputSrc);
                    mesh.material.map = texture;
    
                    const currentMaps = [inputSrc, ''];
                    mapsArray.push(currentMaps);
                    heightmapsArray.push(array.data);
                    const divMapsContainer = document.createElement('div');
                    divMapsContainer.appendChild(img);
                    divMapsContainer.appendChild(textureImg);
                    mapsContainer.appendChild(divMapsContainer);
                    params.currentId = mapsArray.length - 1;
                    const thisMapId = mapsArray.length - 1;
    
                    img.onload = () => {
                        ctx.drawImage(img, 0, 0);
                        const pixels = ctx.getImageData(0, 0, worldWidth, worldDepth).data;
                        const vertices = mesh.geometry.attributes.position.array as Array<number>;
                        for (let i = 0; i < worldWidth * worldWidth; i++) {
                            vertices[1 + i * 3] = pixels[0 + i * 4] * 10
                        }
                        mesh.geometry.attributes.position.needsUpdate = true;
                    }
    
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
                });
            }
        }
    });
    const importHeightmap = () => {
        input.click();
    }
    const buttonImportHeightmap = document.getElementById('btn_import_heightmap') as HTMLElement;
    buttonImportHeightmap.addEventListener('click', importHeightmap);
}
